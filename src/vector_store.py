"""Milvus 向量存储 + 混合检索。

- 纯稠密检索：similarity_search / as_retriever
- 混合检索：hybrid_search（客户端 BM25 + RRF 融合）

BM25 使用 rank_bm25 客户端实现，文本常驻内存，不依赖 Milvus SPARSE_FLOAT_VECTOR。
Milvus 仅存储稠密向量 + 元数据（chunk_id/source/page/section/chunk_type）。
"""

import threading
import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pymilvus import MilvusClient

from src.config import settings
from src.embeddings import create_embeddings
from src.logger import get_logger

logger = get_logger(__name__)


class MilvusStore(VectorStore):
    """Based on pymilvus MilvusClient, supporting dense retrieval + BM25 hybrid search."""

    def __init__(self) -> None:
        logger.info("Connecting Milvus %s:%d", settings.milvus_host, settings.milvus_port)
        self._client = MilvusClient(
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        self._embedding = create_embeddings()
        self._ensure_collection()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with error handling. Raises on failure."""
        try:
            return self._embedding.embed_documents(texts)
        except Exception:
            logger.exception("Embedding failed")
            raise

    def _milvus_insert(self, data: list[dict]) -> None:
        """Insert vectors into Milvus. Raises on failure."""
        try:
            self._client.insert(collection_name=settings.milvus_collection_name, data=data)
        except Exception:
            logger.exception("Milvus insert failed")
            raise

    def _ensure_collection(self) -> None:
        """确保 collection 存在（简单 schema：稠密向量 + 元数据）。"""
        if self._client.has_collection(settings.milvus_collection_name):
            logger.debug("Collection '%s' 已存在", settings.milvus_collection_name)
            return
        logger.info("创建 Collection: %s (dim=%d, metric=COSINE)", settings.milvus_collection_name, settings.embedding_dimension)
        self._client.create_collection(
            collection_name=settings.milvus_collection_name,
            dimension=settings.embedding_dimension,
            metric_type="COSINE",
            auto_id=True,
        )

    @classmethod
    def from_texts(cls, texts: list[str], embedding, metadatas=None, **kwargs):
        store = cls()
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts), strict=False)
        ]
        store.add_documents(docs)
        return store

    def add_documents(self, documents: list[Document], **kwargs) -> list[str]:
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
        embeddings = self._embed(texts)

        from src.database import save_chunk

        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk_id = uuid.uuid4().hex[:16]
            source = chunk.metadata.get("source", "")
            page = chunk.metadata.get("page", 0)
            section = chunk.metadata.get("section_title", "")
            chunk_index = chunk.metadata.get("chunk_index", 0)

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
            milvus_data.append(
                {
                    "vector": emb,
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                    "section": section,
                    "chunk_type": "text",
                }
            )

        self._milvus_insert(milvus_data)

        logger.info("文本块写入完成: %d 条", len(chunk_ids))
        return chunk_ids

    def add_image_placeholders(self, records) -> list[str]:
        """第一阶段：写入图片占位（描述为空），返回 chunk_id 列表。"""
        from src.database import save_chunk, save_image

        if not records:
            return []

        texts = []
        for r in records:
            text = r.caption or f"[图片] {r.source} 第{r.page}页"
            texts.append(text)

        logger.info("写 %d image placeholder(s)", len(texts))
        embeddings = self._embed(texts)

        chunk_ids: list[str] = []

        for record, emb in zip(records, embeddings, strict=False):
            chunk_id = uuid.uuid4().hex[:16]

            save_chunk(
                chunk_id=chunk_id,
                content=record.caption or "",
                source=record.source,
                page=record.page,
                section_title="",
                chunk_index=0,
                chunk_type="image",
            )

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

            self._client.insert(
                collection_name=settings.milvus_collection_name,
                data=[
                    {
                        "vector": emb,
                        "chunk_id": chunk_id,
                        "image_id": record.id,
                        "source": record.source,
                        "page": record.page,
                        "section": "",
                        "chunk_type": "image",
                    }
                ],
            )

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

        for record, emb, text in zip(records, embeddings, texts, strict=False):
            if not record.chunk_id:
                continue
            update_chunk_content(record.chunk_id, text)
            self._client.delete(
                collection_name=settings.milvus_collection_name,
                filter=f'chunk_id == "{record.chunk_id}"',
            )
            self._client.insert(
                collection_name=settings.milvus_collection_name,
                data=[
                    {
                        "vector": emb,
                        "chunk_id": record.chunk_id,
                        "image_id": record.id,
                        "source": record.source,
                        "page": record.page,
                        "section": "",
                        "chunk_type": "image",
                    }
                ],
            )

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
        embeddings = self._embed(texts)

        chunk_ids: list[str] = []
        milvus_data: list[dict] = []

        for i, (record, emb) in enumerate(zip(records, embeddings, strict=False)):
            chunk_id = uuid.uuid4().hex[:16]
            image_id = record.id
            text = texts[i]

            save_chunk(
                chunk_id=chunk_id,
                content=text,
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
            milvus_data.append(
                {
                    "vector": emb,
                    "chunk_id": chunk_id,
                    "image_id": image_id,
                    "source": record.source,
                    "page": record.page,
                    "section": "",
                    "chunk_type": "image",
                }
            )

        self._milvus_insert(milvus_data)

        logger.info("图片描述写入完成: %d 条", len(chunk_ids))
        return chunk_ids

    def delete_all(self) -> None:
        """清空 Milvus 中所有数据（重建 collection）。"""
        if self._client.has_collection(settings.milvus_collection_name):
            self._client.drop_collection(settings.milvus_collection_name)
            logger.info("已删除 collection: %s", settings.milvus_collection_name)
        self._ensure_collection()

    # ═══════════════════════════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════════════════════════

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """纯稠密向量检索。"""
        logger.debug("相似度检索 query='%s...' k=%d", query[:80], k)

        try:
            query_emb = self._embedding.embed_query(query)
        except Exception:
            logger.exception("查询 Embedding 失败")
            raise

        try:
            results = self._client.search(
                collection_name=settings.milvus_collection_name,
                data=[query_emb],
                limit=k,
                output_fields=["chunk_id", "image_id", "source", "page", "section", "chunk_type"],
            )
        except Exception:
            logger.exception("Milvus 检索失败")
            raise

        return self._hits_to_docs(results[0] if results else [])

    def hybrid_search(self, query: str, k: int | None = None, **kwargs) -> list[Document]:
        """BM25 + 稠密向量混合检索，RRF 融合。

        客户端 BM25 (rank_bm25) + Milvus 稠密向量 → RRF 融合。
        """
        k = k or settings.hybrid_retrieval_top_k

        from src.hybrid_search import hybrid_search as do_hybrid

        def dense_fn(q, n):
            return self.similarity_search(q, k=n)

        return do_hybrid(query, dense_fn, k_dense=k * 2, k_sparse=k * 2, k_final=k)

    def _hits_to_docs(self, hits: list) -> list[Document]:
        """将 Milvus 搜索结果转为 LangChain Document 列表。"""
        from src.database import get_chunk

        docs = []
        for hit in hits:
            entity = hit.get("entity", {})
            score = hit.get("distance", 0)
            chunk_id = entity.get("chunk_id", "")
            image_id = entity.get("image_id", "")
            chunk_type = entity.get("chunk_type", "text")
            source = entity.get("source", "")
            page = entity.get("page", 0)
            section = entity.get("section", "")

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

    # ═══════════════════════════════════════════════════════════════
    # 检索器工厂
    # ═══════════════════════════════════════════════════════════════

    def as_retriever(self, **kwargs):
        from langchain_core.retrievers import BaseRetriever

        store = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, **kw):
                k = kw.pop("k", settings.retrieval_top_k)
                return store.similarity_search(query, k=k, **kw)

        return _Retriever()

    def as_hybrid_retriever(self, **kwargs):
        """返回使用混合检索（BM25 + Dense）的 retriever。"""
        from langchain_core.retrievers import BaseRetriever

        store = self

        class _HybridRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, **kw):
                k = kw.pop("k", settings.hybrid_retrieval_top_k)
                return store.hybrid_search(query, k=k, **kw)

        return _HybridRetriever()


# ═══════════════════════════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════════════════════════

_store: MilvusStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> MilvusStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MilvusStore()
    return _store


def add_documents_to_store(documents: list[Document]) -> None:
    """将文档导入知识库：分块 + 图片管线（两阶段：占位→并发生成描述）。"""
    logger.info("收到 %d 个文档待入库", len(documents))
    try:
        store = get_vector_store()

        store.add_documents(documents)

        if settings.enable_image_understanding:
            from src.image_pipeline import fill_image_descriptions, save_image_placeholders

            all_records: list = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                source = doc.metadata.get("source", "")
                if not file_path or not file_path.lower().endswith(".pdf"):
                    if file_path:
                        logger.debug("非 PDF 文件，跳过图片处理: %s", source)
                    continue

                try:
                    records = save_image_placeholders(file_path, source)
                    all_records.extend(records)
                except Exception:
                    logger.exception("图片占位写入失败: %s", source)

            if all_records:
                fill_image_descriptions(all_records)

    except Exception:
        logger.exception("文档入库失败")
        raise


def get_retriever():
    """返回默认 retriever。"""
    store = get_vector_store()
    if settings.hybrid_enabled:
        return store.as_hybrid_retriever()
    return store.as_retriever()


def get_hybrid_retriever():
    """显式返回混合检索器。"""
    return get_vector_store().as_hybrid_retriever()
