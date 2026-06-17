"""检索管线评估脚本。

对比 baseline（纯稠密向量）vs hybrid（BM25 + Dense）vs hybrid + rerank 的检索效果。

用法:
    uv run python scripts/eval_retrieval.py

要求：先执行 ingest 导入 paper1。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.reranker import get_reranker
from src.retrieval_eval import compare_pipelines, load_paper1_queries, run_eval
from src.vector_store import get_vector_store


def baseline_retrieve(query: str, k: int = 5):
    """纯稠密向量检索。"""
    return get_vector_store().similarity_search(query, k=k)


def hybrid_retrieve(query: str, k: int = 5):
    """BM25 + 稠密混合检索（无重排序）。"""
    return get_vector_store().hybrid_search(query, k=20)[:k]


def hybrid_rerank_retrieve(query: str, k: int = 5):
    """混合检索 + FlashRank 重排序。"""
    store = get_vector_store()
    docs = store.hybrid_search(query, k=20)
    reranker = get_reranker()
    return reranker.compress_documents(docs, query)[:k]


def main():
    print("=" * 72)
    print("医疗 RAG 检索管线评估")
    print(f"  hybrid_enabled: {settings.hybrid_enabled}")
    print(f"  reranker_enabled: {settings.reranker_enabled}")
    print(f"  hybrid_retrieval_top_k: {settings.hybrid_retrieval_top_k}")
    print(f"  reranker_top_n: {settings.reranker_top_n}")
    print(f"  reranker_model: {settings.reranker_model}")
    print("=" * 72)

    queries = load_paper1_queries()
    print(f"\n评估用例: {len(queries)} 条\n")

    # ── 单管线评估 ──
    k_values = (5, 10, 20)

    print("─" * 36)
    print("1. Baseline: 纯稠密向量 (COSINE top-5)")
    print("─" * 36)
    baseline = run_eval(
        lambda q: baseline_retrieve(q, k=max(k_values)),
        queries,
        k_values=k_values,
        name="baseline",
    )
    _print_result(baseline, k_values)

    print()
    print("─" * 36)
    print("2. Hybrid: BM25 + Dense RRF (top-20 → top-5)")
    print("─" * 36)
    hybrid = run_eval(
        lambda q: hybrid_retrieve(q, k=max(k_values)),
        queries,
        k_values=k_values,
        name="hybrid",
    )
    _print_result(hybrid, k_values)

    print()
    print("─" * 36)
    print("3. Hybrid + Rerank: BM25 + Dense → FlashRank (top-20→5)")
    print("─" * 36)
    rerank = run_eval(
        lambda q: hybrid_rerank_retrieve(q, k=max(k_values)),
        queries,
        k_values=k_values,
        name="hybrid+rerank",
    )
    _print_result(rerank, k_values)

    # ── 对比报告 ──
    print()
    print("─" * 36)
    print("4. 对比：Baseline vs Hybrid+Rerank")
    print("─" * 36)
    comparison = compare_pipelines(
        lambda q: baseline_retrieve(q, k=5),
        lambda q: hybrid_rerank_retrieve(q, k=5),
        queries,
        k_values=(5,),
    )

    for key, vals in comparison["deltas"].items():
        b = vals["baseline"]
        c = vals["candidate"]
        d = vals["delta"]
        pct = vals.get("pct", "")
        flag = "✅" if d > 0 else ("➖" if d == 0 else "❌")
        print(f"  {key:20s}  baseline={b:.4f}  candidate={c:.4f}  Δ={d:+.4f}  {pct}  {flag}")

    print()
    print("=" * 72)
    print("评估完成。详细 per-query 结果见上方输出。")
    print("=" * 72)


def _print_result(result, k_values):
    """打印单条评估结果。"""
    for k in k_values:
        print(f"  Recall@{k}:   {result.recall_at_k.get(k, 'N/A')}")
    for k in k_values:
        print(f"  Precision@{k}: {result.precision_at_k.get(k, 'N/A')}")
    for k in k_values:
        print(f"  NDCG@{k}:      {result.ndcg_at_k.get(k, 'N/A')}")
    print(f"  MRR:           {result.mrr}")

    # Per-query drill-down
    failures = [q for q in result.per_query if q["relevant@5"] == 0]
    if failures:
        print(f"  ⚠ 零召回查询 ({len(failures)}/{len(result.per_query)}):")
        for f in failures:
            print(f"    - [{f['label']}] {f['query'][:80]}")


if __name__ == "__main__":
    main()
