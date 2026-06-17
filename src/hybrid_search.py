"""混合检索器：BM25 稀疏 + 稠密向量 → RRF 融合。

使用 rank_bm25 做客户端 BM25，不依赖 Milvus SPARSE_FLOAT_VECTOR。
适合中小规模知识库（<10万 chunk），BM25 索引常驻内存。

用法:
    from src.hybrid_search import HybridRetriever
    hr = HybridRetriever(vector_store)
    docs = hr.hybrid_search("query", k=20)
"""

import threading

from langchain_core.documents import Document

from src.logger import get_logger

logger = get_logger(__name__)


class BM25SparseRetriever:
    """客户端 BM25 检索器，所有 chunk 文本常驻内存。"""

    def __init__(self):
        self._chunks: list[dict] = []  # [{id, text, metadata}, ...]
        self._bm25 = None

    def _build_index(self) -> None:
        """从 PG 加载所有文本 chunk 并构建 BM25 索引。"""
        from rank_bm25 import BM25Okapi
        from sqlalchemy import text

        from src.database import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, content, source, page, section_title, chunk_type "
                    "FROM chunks WHERE chunk_type = 'text' ORDER BY created_at"
                )
            ).fetchall()

        self._chunks = [dict(r._mapping) for r in rows]
        if not self._chunks:
            logger.warning("BM25: PG 中无文本 chunk")
            self._bm25 = None
            return

        # 分词：按空格简单切分（英文医学文本适用）
        tokenized = [(c["content"] or "").lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 索引构建: %d 个 chunk", len(self._chunks))

    def search(self, query: str, k: int = 20) -> list[Document]:
        """BM25 检索，返回带分数的 Document 列表。"""
        if self._bm25 is None:
            self._build_index()
        if self._bm25 is None or not self._chunks:
            return []

        tokens = (query or "").lower().split()
        scores = self._bm25.get_scores(tokens)
        # 取 top-k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        docs = []
        for idx, score in indexed:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            docs.append(
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "chunk_id": chunk["id"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "section": chunk.get("section_title", ""),
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "bm25_score": round(float(score), 4),
                    },
                )
            )

        return docs


_bm25_retriever: BM25SparseRetriever | None = None
_bm25_lock = threading.Lock()


def get_bm25_retriever() -> BM25SparseRetriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        with _bm25_lock:
            if _bm25_retriever is None:
                _bm25_retriever = BM25SparseRetriever()
    return _bm25_retriever


def _rrf_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k_rrf: int = 60,
    top_k: int = 20,
) -> list[Document]:
    """Reciprocal Rank Fusion 合并两个检索结果列表。"""
    scores: dict[str, tuple[float, Document]] = {}

    for rank, doc in enumerate(dense_docs):
        cid = doc.metadata.get("chunk_id", "")
        if not cid:
            continue
        rrf = 1.0 / (k_rrf + rank + 1)
        scores[cid] = (rrf, doc)

    for rank, doc in enumerate(sparse_docs):
        cid = doc.metadata.get("chunk_id", "")
        if not cid:
            continue
        rrf = 1.0 / (k_rrf + rank + 1)
        if cid in scores:
            prev, existing = scores[cid]
            scores[cid] = (prev + rrf, existing)
        else:
            scores[cid] = (rrf, doc)

    ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    result = []
    for rrf_score, doc in ranked[:top_k]:
        doc.metadata["rrf_score"] = round(float(rrf_score), 6)
        result.append(doc)

    return result


def hybrid_search(
    query: str,
    dense_fn,
    k_dense: int = 20,
    k_sparse: int = 20,
    k_final: int = 20,
) -> list[Document]:
    """执行 BM25 + 稠密混合检索。

    Args:
        query: 查询文本
        dense_fn: callable(query, k) → list[Document] 稠密检索函数
        k_dense: 稠密检索数量
        k_sparse: BM25 检索数量
        k_final: 最终返回数量

    Returns:
        RRF 融合后的文档列表
    """
    bm25 = get_bm25_retriever()

    dense_docs = dense_fn(query, k_dense)
    sparse_docs = bm25.search(query, k_sparse)

    logger.debug(
        "混合检索: dense=%d sparse=%d → RRF fusion → %d",
        len(dense_docs),
        len(sparse_docs),
        min(k_final, len(dense_docs) + len(sparse_docs)),
    )

    return _rrf_fusion(dense_docs, sparse_docs, top_k=k_final)
