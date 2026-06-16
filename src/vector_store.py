import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pymilvus import MilvusClient

from src.config import settings
from src.embeddings import create_embeddings
from src.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "hospital_knowledge"
DIMENSION = 1024


class MilvusStore(VectorStore):
    """基于 pymilvus MilvusClient 的向量存储，不依赖 ORM API。"""

    def __init__(self) -> None:
        logger.info("连接 Milvus %s:%d", settings.milvus_host, settings.milvus_port)
        self._client = MilvusClient(
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        self._embedding = create_embeddings()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.has_collection(COLLECTION_NAME):
            logger.debug("Collection '%s' 已存在", COLLECTION_NAME)
            return
        logger.info("创建 Collection: %s (dim=%d, metric=COSINE)", COLLECTION_NAME, DIMENSION)
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            metric_type="COSINE",
            auto_id=True,
        )

    @classmethod
    def from_texts(cls, texts: list[str], embedding, metadatas=None, **kwargs):
        store = cls()
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        store.add_documents(docs)
        return store

    def add_documents(
        self, documents: list[Document], **kwargs
    ) -> list[str]:
        """分块 → 写 PG → Embed → 写 Milvus。返回 chunk_id 列表。"""
        from src.text_splitter import create_semantic_splitter

        splitter = create_semantic_splitter()
        chunks = splitter(documents)
        if not chunks:
            logger.warning("文档分块后为空，跳过写入")
            return []

        logger.info("开始写入: %d 个文本块", len(chunks))
        chunk_ids: list[str] = []
        milvus_data: list[dict] = []

        texts = [chunk.page_content for chunk in chunks]
        try:
            embeddings = self._embedding.embed_documents(texts)
        except Exception:
            logger.exception("Embedding 失败")
            raise

        from src.database import save_chunk

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = uuid.uuid4().hex[:16]
            source = chunk.metadata.get("source", "")
            page = chunk.metadata.get("page", 0) or 0  # opendataloader 用 num_pages，这里统一处理
            section = chunk.metadata.get("section_title", "")
            chunk_index = chunk.metadata.get("chunk_index", 0)

            # 写 PG
            save_chunk(
                chunk_id=chunk_id,
                content=chunk.page_content,
                source=source,
                page=page,
                section_title=section,
                chunk_index=chunk_index,
                chunk_type="text",
            )

            chunk_ids.append(chunk_id)
            milvus_data.append({
                "vector": emb,
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "section": section,
                "chunk_type": "text",
            })

        try:
            self._client.insert(collection_name=COLLECTION_NAME, data=milvus_data)
        except Exception:
            logger.exception("Milvus 写入失败")
            raise

        logger.info("文本块写入完成: %d 条", len(chunk_ids))
        return chunk_ids

    def add_image_placeholders(self, records) -> list[str]:
        """第一阶段：写入图片占位（描述为空），返回 chunk_id 列表。"""
        from src.database import save_chunk, save_image

        if not records:
            return []

        # 用 caption 作为临时向量文本
        texts = []
        for r in records:
            text = r.caption or f"[图片] {r.source} 第{r.page}页"
            texts.append(text)

        logger.info("写入 %d 张图片占位", len(texts))
        try:
            embeddings = self._embedding.embed_documents(texts)
        except Exception:
            logger.exception("图片占位 Embedding 失败")
            raise

        chunk_ids: list[str] = []

        for record, emb in zip(records, embeddings):
            chunk_id = uuid.uuid4().hex[:16]

            # PG chunks 表
            save_chunk(
                chunk_id=chunk_id,
                content=record.caption or "",
                source=record.source,
                page=record.page,
                section_title="",
                chunk_index=0,
                chunk_type="image",
            )

            # PG images 表（含 bbox）
            save_image(
                image_id=record.id,
                chunk_id=chunk_id,
                image_data=record.image_data,
                description="",
                caption=record.caption,
                source=record.source,
                page=record.page,
                image_format=record.image_format,
                bbox=record.bbox,
            )

            record.chunk_id = chunk_id
            chunk_ids.append(chunk_id)

            self._client.insert(collection_name=COLLECTION_NAME, data=[{
                "vector": emb,
                "chunk_id": chunk_id,
                "image_id": record.id,
                "source": record.source,
                "page": record.page,
                "section": "",
                "chunk_type": "image",
            }])

        logger.info("图片占位写入完成: %d 条", len(chunk_ids))
        return chunk_ids

    def update_image_vectors(self, records) -> None:
        """第二阶段：用 qwen 描述更新 Milvus 向量 + PG 内容。"""
        from src.database import update_chunk_content

        texts = []
        for r in records:
            parts = [r.description]
            if r.caption:
                parts.append(f"[图片说明: {r.caption}]")
            texts.append("\n".join(parts))

        try:
            embeddings = self._embedding.embed_documents(texts)
        except Exception:
            logger.exception("图片向量更新 Embedding 失败")
            return

        for record, emb, text in zip(records, embeddings, texts):
            if not record.chunk_id:
                continue
            update_chunk_content(record.chunk_id, text)
            # Milvus 不支持 update vector，删除旧 + 插入新
            self._client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'chunk_id == "{record.chunk_id}"',
            )
            self._client.insert(collection_name=COLLECTION_NAME, data=[{
                "vector": emb,
                "chunk_id": record.chunk_id,
                "image_id": record.id,
                "source": record.source,
                "page": record.page,
                "section": "",
                "chunk_type": "image",
            }])

    def add_image_descriptions(self, records) -> list[str]:
        """（兼容旧接口）将图片描述向量化后写入 Milvus + PG images 表。"""
        from src.database import save_chunk, save_image

        if not records:
            return []

        texts = []
        for r in records:
            parts = [r.description]
            if r.caption:
                parts.append(f"[图片说明: {r.caption}]")
            texts.append("\n".join(parts))

        logger.info("开始 Embedding: %d 个图片描述", len(texts))
        try:
            embeddings = self._embedding.embed_documents(texts)
        except Exception:
            logger.exception("图片描述 Embedding 失败")
            raise

        chunk_ids: list[str] = []
        milvus_data: list[dict] = []

        for record, emb in zip(records, embeddings):
            chunk_id = uuid.uuid4().hex[:16]
            image_id = record.id

            save_chunk(
                chunk_id=chunk_id,
                content=texts[records.index(record)],
                source=record.source,
                page=record.page,
                section_title="",
                chunk_index=0,
                chunk_type="image",
            )

            save_image(
                image_id=image_id,
                chunk_id=chunk_id,
                image_data=record.image_data,
                description=record.description,
                caption=record.caption,
                source=record.source,
                page=record.page,
                image_format=record.image_format,
                bbox=record.bbox,
            )

            chunk_ids.append(chunk_id)
            milvus_data.append({
                "vector": emb,
                "chunk_id": chunk_id,
                "image_id": image_id,
                "source": record.source,
                "page": record.page,
                "section": "",
                "chunk_type": "image",
            })

        try:
            self._client.insert(collection_name=COLLECTION_NAME, data=milvus_data)
        except Exception:
            logger.exception("Milvus 图片写入失败")
            raise

        logger.info("图片描述写入完成: %d 条", len(chunk_ids))
        return chunk_ids

    def delete_all(self) -> None:
        """清空 Milvus 中所有数据（重建 collection）。"""
        if self._client.has_collection(COLLECTION_NAME):
            self._client.drop_collection(COLLECTION_NAME)
            logger.info("已删除 collection: %s", COLLECTION_NAME)
        self._ensure_collection()

    def similarity_search(
        self, query: str, k: int = 5, **kwargs
    ) -> list[Document]:
        logger.debug("检索 query='%s...' k=%d", query[:80], k)

        try:
            query_emb = self._embedding.embed_query(query)
        except Exception:
            logger.exception("查询 Embedding 失败")
            raise

        try:
            results = self._client.search(
                collection_name=COLLECTION_NAME,
                data=[query_emb],
                limit=k,
                output_fields=["chunk_id", "image_id", "source", "page", "section", "chunk_type"],
            )
        except Exception:
            logger.exception("Milvus 检索失败")
            raise

        docs = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            score = hit.get("distance", 0)
            chunk_id = entity.get("chunk_id", "")
            image_id = entity.get("image_id", "")
            chunk_type = entity.get("chunk_type", "text")
            source = entity.get("source", "")
            page = entity.get("page", 0)
            section = entity.get("section", "")

            logger.debug("  hit: %s P%s type=%s score=%.4f id=%s", source, page, chunk_type, score, chunk_id)

            # 从 PG 读取原文
            from src.database import get_chunk

            chunk = get_chunk(chunk_id)
            content = chunk["content"] if chunk else ""
            source = chunk["source"] if chunk else source

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "page": page,
                        "section": section,
                        "chunk_id": chunk_id,
                        "image_id": image_id,
                        "chunk_type": chunk_type,
                        "score": round(score, 4),
                    },
                )
            )

        logger.info("检索完成: 返回 %d 个文档块", len(docs))
        return docs

    def as_retriever(self, **kwargs):
        from langchain_core.retrievers import BaseRetriever

        store = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, **kw):
                k = kw.pop("k", settings.retrieval_top_k)
                return store.similarity_search(query, k=k, **kw)

        return _Retriever()


_store: MilvusStore | None = None


def get_vector_store() -> MilvusStore:
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store


def add_documents_to_store(documents: list[Document]) -> None:
    """将文档导入知识库：分块 + 图片管线（两阶段：占位→并发生成描述）。"""
    logger.info("收到 %d 个文档待入库", len(documents))
    try:
        store = get_vector_store()

        # 1. 文本分块入库
        store.add_documents(documents)

        # 2. 图片管线：第一阶段 — 提取 + 写占位
        if settings.enable_image_understanding:
            from src.image_pipeline import save_image_placeholders, fill_image_descriptions

            all_records: list = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                source = doc.metadata.get("source", "")
                if not file_path:
                    logger.warning("文档缺少 file_path，跳过图片处理: %s", source)
                    continue

                try:
                    records = save_image_placeholders(file_path, source)
                    all_records.extend(records)
                except Exception:
                    logger.exception("图片占位写入失败: %s", source)

            # 第二阶段：并发生成描述 + UPDATE
            if all_records:
                fill_image_descriptions(all_records)

    except Exception:
        logger.exception("文档入库失败")
        raise


def get_retriever():
    store = get_vector_store()
    return store.as_retriever()
