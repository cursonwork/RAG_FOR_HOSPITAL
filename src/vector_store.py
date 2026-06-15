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
        from src.text_splitter import create_text_splitter

        splitter = create_text_splitter()
        chunks = splitter.split_documents(documents)
        if not chunks:
            logger.warning("文档分块后为空，跳过写入")
            return []

        logger.info("开始 Embedding: %d 个文本块", len(chunks))
        texts = [chunk.page_content for chunk in chunks]

        try:
            embeddings = self._embedding.embed_documents(texts)
        except Exception:
            logger.exception("Embedding 失败")
            raise

        data = []
        for chunk, emb in zip(chunks, embeddings):
            data.append({
                "vector": emb,
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", 0),
            })

        logger.info("写入 Milvus: %d 条向量", len(data))
        try:
            self._client.insert(collection_name=COLLECTION_NAME, data=data)
        except Exception:
            logger.exception("Milvus 写入失败")
            raise

        logger.info("写入完成，共 %d 条", len(data))
        return []

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
                output_fields=["text", "source", "page"],
            )
        except Exception:
            logger.exception("Milvus 检索失败")
            raise

        docs = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            score = hit.get("distance", 0)
            source = entity.get("source", "")
            page = entity.get("page", 0)
            logger.debug("  hit: %s 第%s页 score=%.4f", source, page, score)
            docs.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata={"source": source, "page": page},
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
    logger.info("收到 %d 个文档待入库", len(documents))
    try:
        store = get_vector_store()
        store.add_documents(documents)
    except Exception:
        logger.exception("文档入库失败")
        raise


def get_retriever():
    store = get_vector_store()
    return store.as_retriever()
