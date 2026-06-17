#!/usr/bin/env python3
"""RAG 系统全面评估脚本。

运行完整的检索 + 生成 + 端到端评估，生成详细报告。

用法:
    # 仅检索评估（离线，不需要 LLM）
    uv run python scripts/run_full_evaluation.py --retrieval-only

    # 完整评估（包含 LLM-as-judge，需要 DEEPSEEK_API_KEY）
    uv run python scripts/run_full_evaluation.py

    # 管线对比
    uv run python scripts/run_full_evaluation.py --compare

    # 切片分析
    uv run python scripts/run_full_evaluation.py --slices
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logger import get_logger
from src.reranker import get_reranker
from src.vector_store import get_vector_store

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 检索管线工厂函数
# ═══════════════════════════════════════════════════════════════


def baseline_retrieve(query: str, k: int = 20):
    """纯稠密向量检索。"""
    return get_vector_store().similarity_search(query, k=k)


def hybrid_retrieve(query: str, k: int = 20):
    """BM25 + 稠密混合检索。"""
    return get_vector_store().hybrid_search(query, k=k)


def hybrid_rerank_retrieve(query: str, k: int = 5):
    """混合检索 + FlashRank 重排序。"""
    docs = get_vector_store().hybrid_search(query, k=settings.hybrid_retrieval_top_k)
    reranker = get_reranker()
    try:
        return reranker.compress_documents(docs, query)[:k]
    except Exception:
        # 重排序不可用时降级为去重截断
        seen = set()
        unique = []
        for d in docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique[:k]


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="RAG 系统全面评估")
    parser.add_argument("--retrieval-only", action="store_true", help="仅运行检索评估（不需要 LLM）")
    parser.add_argument("--compare", action="store_true", help="对比多条检索管线")
    parser.add_argument("--slices", action="store_true", help="按维度切片分析")
    parser.add_argument("--output-dir", default="data/eval_results", help="评估结果输出目录")
    args = parser.parse_args()

    from src.evaluation.dataset import dataset_stats, get_dataset
    from src.evaluation.metrics import compute_retrieval_metrics, compute_slice_metrics
    from src.evaluation.runner import (
        EvalConfig,
        generate_report,
        run_full_evaluation,
    )

    print("=" * 72)
    print("  RAG 系统全面评估 — 医疗知识库")
    print("=" * 72)
    stats = dataset_stats()
    print(f"\n数据集: {stats['total']} 条 QA 对 (7 篇论文)")
    print(f"  类型: {stats['by_type']}")
    print(f"  难度: {stats['by_difficulty']}")
    print("\n系统配置:")
    print(f"  HYBRID_ENABLED: {settings.hybrid_enabled}")
    print(f"  RERANKER_ENABLED: {settings.reranker_enabled}")
    print(f"  RERANKER_MODEL: {settings.reranker_model}")
    print(f"  HYBRID_RETRIEVAL_TOP_K: {settings.hybrid_retrieval_top_k}")
    print(f"  RETRIEVAL_TOP_K: {settings.retrieval_top_k}")
    print()

    queries = get_dataset()
    k_values = (1, 3, 5, 10, 20)

    # ── 管线对比 ──
    if args.compare:
        print("=" * 72)
        print("  检索管线对比: Baseline vs Hybrid vs Hybrid+Rerank")
        print("=" * 72)
        print()

        pipelines = {
            "baseline (Dense only)": baseline_retrieve,
            "hybrid (BM25 + Dense)": hybrid_retrieve,
            "hybrid + rerank": hybrid_rerank_retrieve,
        }

        results = {}
        for name, fn in pipelines.items():
            print(f"评估: {name}...")
            t0 = time.perf_counter()
            results[name] = compute_retrieval_metrics(fn, queries, k_values=k_values)
            t1 = time.perf_counter()
            print(f"  完成 ({t1 - t0:.1f}s)")

        # ── 打印对比表 ──
        print()
        print("┌─ 管线对比 ─ @5 指标")
        print("│")
        header = f"│ {'指标':<20}"
        for name in pipelines:
            header += f" {name:<32}"
        print(header)
        print("│ " + "-" * (20 + 32 * 3))

        for metric_key, metric_label in [
            ("recall_at_k", "Recall@5"),
            ("precision_at_k", "Precision@5"),
            ("ndcg_at_k", "NDCG@5"),
            ("hit_at_k", "Hit Rate@5"),
        ]:
            row = f"│ {metric_label:<20}"
            best_val = -1
            _best_name = ""
            for name in pipelines:
                val = getattr(results[name], metric_key).get(5, 0)
                row += f" {val:<32.4f}"
                if val > best_val:
                    best_val = val
                    _best_name = name
            print(row)

        # MRR and MAP
        for metric_key, metric_label in [("mrr", "MRR"), ("map_score", "MAP")]:
            row = f"│ {metric_label:<20}"
            for name in pipelines:
                val = getattr(results[name], metric_key)
                row += f" {val:<32.4f}"
            print(row)

        print("│")
        print("└─")

        # Per-paper breakdown for best pipeline
        print()
        best_pipeline = "hybrid + rerank"
        print(f"┌─ {best_pipeline} 逐论文分析 (@5)")
        print("│")
        slice_result = compute_slice_metrics(hybrid_rerank_retrieve, queries, k_value=5)
        for slice_name in sorted(slice_result.keys()):
            if slice_name.startswith("paper:"):
                s = slice_result[slice_name]
                print(
                    f"│  {slice_name:<30}  n={s['count']:>3}  "
                    f"R@5={s['recall@5']:.4f}  P@5={s['precision@5']:.4f}  "
                    f"NDCG@5={s['ndcg@5']:.4f}  MRR={s['mrr']:.4f}"
                )
        print("│")
        print("└─")

        # By question type
        print()
        print(f"┌─ {best_pipeline} 逐问题类型分析 (@5)")
        print("│")
        for slice_name in sorted(slice_result.keys()):
            if slice_name.startswith("type:"):
                s = slice_result[slice_name]
                print(
                    f"│  {slice_name:<30}  n={s['count']:>3}  "
                    f"R@5={s['recall@5']:.4f}  P@5={s['precision@5']:.4f}  "
                    f"MRR={s['mrr']:.4f}"
                )
        print("│")
        print("└─")

        # By difficulty
        print()
        print(f"┌─ {best_pipeline} 逐难度分析 (@5)")
        print("│")
        for slice_name in sorted(slice_result.keys()):
            if slice_name.startswith("difficulty:"):
                s = slice_result[slice_name]
                print(
                    f"│  {slice_name:<30}  n={s['count']:>3}  "
                    f"R@5={s['recall@5']:.4f}  P@5={s['precision@5']:.4f}  "
                    f"MRR={s['mrr']:.4f}"
                )
        print("│")
        print("└─")

        # Print per-query failures
        print()
        best_result = results[best_pipeline]
        failures_0 = [q for q in best_result.per_query if q["relevant@5"] == 0]
        if failures_0:
            print(f"⚠ 零召回查询 ({len(failures_0)}/{len(best_result.per_query)}):")
            for f in failures_0:
                print(f"  - [{f['id']}] [{f['paper'][:30]}] [{f['type']}] {f['query'][:80]}")
        else:
            print("✅ 无零召回查询！")

        # DELTA report: baseline vs hybrid+rerank
        print()
        print("─" * 72)
        print("Baseline vs Hybrid+Rerank 提升幅度 (@5):")
        b = results["baseline (Dense only)"]
        c = results["hybrid + rerank"]
        for metric_key, metric_label in [
            ("recall_at_k", "Recall@5"),
            ("precision_at_k", "Precision@5"),
            ("ndcg_at_k", "NDCG@5"),
            ("hit_at_k", "Hit Rate@5"),
        ]:
            bv = getattr(b, metric_key).get(5, 0)
            cv = getattr(c, metric_key).get(5, 0)
            delta = cv - bv
            pct = f"{delta / bv * 100:+.1f}%" if bv > 0 else "N/A"
            flag = "✅" if delta > 0 else ("➖" if delta == 0 else "❌")
            print(f"  {metric_label:<20}  {bv:.4f} → {cv:.4f}  Δ={delta:+.4f}  {pct}  {flag}")

        b_mrr = b.mrr
        c_mrr = c.mrr
        delta_mrr = c_mrr - b_mrr
        pct_mrr = f"{delta_mrr / b_mrr * 100:+.1f}%" if b_mrr > 0 else "N/A"
        print(f"  {'MRR':<20}  {b_mrr:.4f} → {c_mrr:.4f}  Δ={delta_mrr:+.4f}  {pct_mrr}")
        print("─" * 72)

        return

    # ── 切片分析（独立模式）──
    if args.slices:
        print("切片分析（Hybrid+Rerank 管线）...\n")
        slice_result = compute_slice_metrics(hybrid_rerank_retrieve, queries, k_value=5)

        for group_name in ["paper", "type", "difficulty"]:
            print(f"\n┌─ 按 {group_name} 切片")
            for slice_name in sorted(slice_result.keys()):
                if slice_name.startswith(f"{group_name}:"):
                    s = slice_result[slice_name]
                    print(
                        f"│  {slice_name:<35}  n={s['count']:>3}  "
                        f"R@5={s['recall@5']:.4f}  P@5={s['precision@5']:.4f}  "
                        f"NDCG@5={s['ndcg@5']:.4f}  MRR={s['mrr']:.4f}"
                    )
            print("└─")
        return

    # ── 标准评估 ──
    if args.retrieval_only:
        print("运行模式: 仅检索评估\n")

        pipelines = [
            ("1. Baseline (Dense only)", baseline_retrieve),
            ("2. Hybrid (BM25 + Dense)", hybrid_retrieve),
            ("3. Hybrid + Rerank", hybrid_rerank_retrieve),
        ]

        for label, fn in pipelines:
            print(f"\n{'─' * 72}")
            print(f" {label}")
            print(f"{'─' * 72}")
            t0 = time.perf_counter()
            metrics = compute_retrieval_metrics(fn, queries, k_values=k_values)
            t1 = time.perf_counter()

            print(
                f"  Recall@1/3/5/10/20:  "
                f"{metrics.recall_at_k.get(1, 0):.4f} / {metrics.recall_at_k.get(3, 0):.4f} / "
                f"{metrics.recall_at_k.get(5, 0):.4f} / {metrics.recall_at_k.get(10, 0):.4f} / "
                f"{metrics.recall_at_k.get(20, 0):.4f}"
            )
            print(
                f"  Precision@1/3/5/10/20: "
                f"{metrics.precision_at_k.get(1, 0):.4f} / {metrics.precision_at_k.get(3, 0):.4f} / "
                f"{metrics.precision_at_k.get(5, 0):.4f} / {metrics.precision_at_k.get(10, 0):.4f} / "
                f"{metrics.precision_at_k.get(20, 0):.4f}"
            )
            print(
                f"  NDCG@1/3/5/10/20:     "
                f"{metrics.ndcg_at_k.get(1, 0):.4f} / {metrics.ndcg_at_k.get(3, 0):.4f} / "
                f"{metrics.ndcg_at_k.get(5, 0):.4f} / {metrics.ndcg_at_k.get(10, 0):.4f} / "
                f"{metrics.ndcg_at_k.get(20, 0):.4f}"
            )
            print(
                f"  Hit Rate@1/3/5/10/20: "
                f"{metrics.hit_at_k.get(1, 0):.4f} / {metrics.hit_at_k.get(3, 0):.4f} / "
                f"{metrics.hit_at_k.get(5, 0):.4f} / {metrics.hit_at_k.get(10, 0):.4f} / "
                f"{metrics.hit_at_k.get(20, 0):.4f}"
            )
            print(f"  MRR: {metrics.mrr:.4f}  (95% CI: {metrics.mrr_ci[0]:.4f}–{metrics.mrr_ci[1]:.4f})")
            print(f"  MAP: {metrics.map_score:.4f}")

            # Quick diagnostics
            failures = [q for q in metrics.per_query if q["relevant@5"] == 0]
            if failures:
                print(f"  ⚠ 零召回@5: {len(failures)}/{len(metrics.per_query)}")
            else:
                print("  ✅ 所有查询至少找回 1 个相关文档")

            print(f"  耗时: {t1 - t0:.1f}s")

        return

    # ── 全评估模式（含 LLM-as-judge）──
    print("运行模式: 完整评估（检索 + LLM-as-judge 生成评估）\n")

    # 创建 LLM
    from src.embeddings import create_embeddings
    from src.llm import create_llm

    try:
        llm = create_llm()
        emb = create_embeddings()
        print("LLM 连接成功，将运行完整评估")
    except Exception:
        print("⚠ LLM 不可用，回退到仅检索评估")
        llm = None
        emb = None

    # 选最优管线
    retrieve_fn = hybrid_rerank_retrieve if settings.hybrid_enabled else baseline_retrieve

    if llm:
        # 创建生成函数
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        from src.prompts import get_system_prompt
        from src.rag_chain import format_docs

        prompt_tpl = ChatPromptTemplate.from_messages(
            [
                ("system", get_system_prompt("medical_qa")),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
            ]
        )
        generate_chain = prompt_tpl | llm | StrOutputParser()

        def generate_fn(question: str, docs) -> str:
            ctx = format_docs(docs)
            return generate_chain.invoke({"context": ctx, "question": question})

        config = EvalConfig(
            k_values=k_values,
            enable_generation_eval=True,
            generation_sample_size=20,
            enable_e2e_eval=True,
            enable_slice_analysis=True,
            output_dir=args.output_dir,
        )

        result = run_full_evaluation(
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            llm=llm,
            embeddings=emb,
            config=config,
            pipeline_name="hybrid+rerank",
        )
    else:
        config = EvalConfig(
            k_values=k_values,
            enable_generation_eval=False,
            enable_e2e_eval=False,
            enable_slice_analysis=True,
            output_dir=args.output_dir,
        )

        result = run_full_evaluation(
            retrieve_fn=retrieve_fn,
            config=config,
            pipeline_name="hybrid+rerank",
        )

    # ── 切片分析 ──
    slices = compute_slice_metrics(retrieve_fn, queries, k_value=5)

    # ── 打印报告 ──
    generate_report(result, output_dir=args.output_dir)

    # ── 切片分析 ──
    print()
    print("┌─ 切片分析 (Slice Analysis)")
    for group_name, group_label in [("paper", "论文"), ("type", "问题类型"), ("difficulty", "难度")]:
        print("│")
        print(f"│  ── 按{group_label} ──")
        for slice_name in sorted(slices.keys()):
            if slice_name.startswith(f"{group_name}:"):
                s = slices[slice_name]
                print(
                    f"│  {slice_name:<35}  n={s['count']:>3}  "
                    f"R@5={s['recall@5']:.4f}  P@5={s['precision@5']:.4f}  "
                    f"NDCG@5={s['ndcg@5']:.4f}  MRR={s['mrr']:.4f}"
                )
    print("│")
    print("└─")

    print()
    print("=" * 72)
    print("  评估完成！")
    print(f"  详细结果: {args.output_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
