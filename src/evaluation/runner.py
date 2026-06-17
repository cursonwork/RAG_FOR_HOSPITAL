"""RAG 系统全面评估运行器。

编排完整的评估流程：检索评估 → 生成评估 → 端到端评估 → 报告生成。
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.dataset import dataset_stats, get_dataset
from src.evaluation.metrics import (
    FullEvalResult,
    compute_e2e_metrics,
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """评估配置。"""

    k_values: tuple[int, ...] = (1, 3, 5, 10, 20)
    enable_generation_eval: bool = True  # 是否运行 LLM-as-judge
    generation_sample_size: int = 20  # LLM 评估的采样数
    enable_e2e_eval: bool = True
    enable_slice_analysis: bool = True
    output_dir: str = "data/eval_results"


def run_full_evaluation(
    retrieve_fn,
    generate_fn=None,
    llm=None,
    embeddings=None,
    config: EvalConfig | None = None,
    pipeline_name: str = "default",
) -> FullEvalResult:
    """运行完整的 RAG 评估。

    Args:
        retrieve_fn: callable(query_str) → list[Document]
        generate_fn: callable(query_str, docs) → str（可选，用于生成评估）
        llm: ChatOpenAI 实例（用于 LLM-as-judge 评估）
        embeddings: Embeddings 实例（用于语义相似度）
        config: 评估配置
        pipeline_name: 管线名称

    Returns:
        FullEvalResult 包含所有指标
    """
    config = config or EvalConfig()
    queries = get_dataset()
    result = FullEvalResult(pipeline_name=pipeline_name)
    result.dataset_stats = dataset_stats()

    logger.info("=" * 60)
    logger.info("RAG 全系统评估开始 — 管线: %s", pipeline_name)
    logger.info("数据集: %d 条 QA 对", len(queries))
    logger.info("=" * 60)

    # ── Phase 1: 检索评估 ──
    logger.info("\n[Phase 1/3] 检索指标评估...")
    t0 = time.perf_counter()
    result.retrieval = compute_retrieval_metrics(retrieve_fn, queries, k_values=config.k_values)
    t1 = time.perf_counter()
    logger.info("检索评估完成 (%.1fs)", t1 - t0)

    # ── Phase 2: 生成 + 端到端评估（需要实际生成回答）──
    answers = []
    contexts = []

    if generate_fn and llm:
        logger.info("\n[Phase 2/3] 生成回答采样（LLM-as-judge 评估）...")
        # 分层采样
        if len(queries) > config.generation_sample_size:
            step = max(1, len(queries) // config.generation_sample_size)
            gen_indices = list(range(0, len(queries), step))[: config.generation_sample_size]
        else:
            gen_indices = list(range(len(queries)))

        gen_queries = [queries[i] for i in gen_indices]

        for i, item in enumerate(gen_queries):
            try:
                docs = retrieve_fn(item.question)
                answer = generate_fn(item.question, docs)
                ctx = "\n---\n".join(
                    f"[{d.metadata.get('source', '?')}:{d.metadata.get('page', '?')}] {d.page_content[:500]}"
                    for d in docs[:5]
                )
                answers.append(answer)
                contexts.append(ctx)
            except Exception:
                logger.warning("生成失败: %s", item.id)
                answers.append("")
                contexts.append("")

            if (i + 1) % 5 == 0:
                logger.info("  已生成 %d/%d 个回答", i + 1, len(gen_queries))

        # 端到端指标
        if config.enable_e2e_eval:
            logger.info("\n[Phase 3/3] 端到端指标评估...")
            result.e2e = compute_e2e_metrics(gen_queries, answers, embeddings)

        # 生成质量指标（LLM-as-judge）
        if config.enable_generation_eval and llm:
            logger.info("\n[Phase 3/3] LLM-as-judge 生成质量评估...")
            result.generation = compute_generation_metrics(
                llm,
                gen_queries,
                answers,
                contexts,
                sample_size=config.generation_sample_size,
            )
    else:
        logger.info("\n[Phase 2-3/3] 跳过（未提供 generate_fn 或 llm）")

    t2 = time.perf_counter()
    logger.info("全系统评估完成 (总耗时 %.1fs)", t2 - t0)

    return result


def generate_report(
    result: FullEvalResult,
    output_dir: str = "data/eval_results",
    save_json: bool = True,
) -> str:
    """生成并打印可读的评估报告，可选保存为 JSON。

    Returns:
        报告文本字符串
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    r = result.retrieval
    g = result.generation
    e = result.e2e

    lines = []

    def _add(line=""):
        lines.append(line)

    _add("=" * 72)
    _add(f"  RAG 系统全面评估报告 — {result.pipeline_name}")
    _add("=" * 72)

    # ── 数据集概览 ──
    _add()
    _add("┌─ 数据集概览")
    stats = result.dataset_stats
    _add(f"│  总 QA 对: {stats['total']}")
    _add(f"│  论文分布: {json.dumps(stats.get('by_paper', {}), ensure_ascii=False)}")
    _add(f"│  类型分布: {stats.get('by_type', {})}")
    _add(f"│  难度分布: {stats.get('by_difficulty', {})}")
    _add("└─")

    # ── 检索指标 ──
    _add()
    _add("┌─ 检索指标 (Retrieval Quality)")
    _add("│")
    _add("│  ┌────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    _add("│  │ 指标               │    @1    │    @3    │    @5    │   @10    │   @20    │")
    _add("│  ├────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    k_vals = [1, 3, 5, 10, 20]

    recall_row = "│  │ Recall              "
    for k in k_vals:
        v = r.recall_at_k.get(k, 0)
        recall_row += f"│  {v:.4f}  "
    recall_row += "│"
    _add(recall_row)

    prec_row = "│  │ Precision           "
    for k in k_vals:
        v = r.precision_at_k.get(k, 0)
        prec_row += f"│  {v:.4f}  "
    prec_row += "│"
    _add(prec_row)

    ndcg_row = "│  │ NDCG                "
    for k in k_vals:
        v = r.ndcg_at_k.get(k, 0)
        ndcg_row += f"│  {v:.4f}  "
    ndcg_row += "│"
    _add(ndcg_row)

    hit_row = "│  │ Hit Rate            "
    for k in k_vals:
        v = r.hit_at_k.get(k, 0)
        hit_row += f"│  {v:.4f}  "
    hit_row += "│"
    _add(hit_row)

    _add("│  └────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    _add(f"│  MRR:  {r.mrr:.4f}  (95% CI: {r.mrr_ci[0]:.4f}–{r.mrr_ci[1]:.4f})")
    _add(f"│  MAP:  {r.map_score:.4f}")
    _add("└─")

    # ── 检索失败诊断 ──
    failures_0 = [q for q in r.per_query if q["relevant@5"] == 0]
    failures_1 = [q for q in r.per_query if q["relevant@5"] <= 1]
    _add()
    _add("┌─ 检索诊断")
    _add(
        f"│  零召回 (relevant@5=0): {len(failures_0)}/{len(r.per_query)} ({len(failures_0) / max(len(r.per_query), 1) * 100:.1f}%)"
    )
    _add(
        f"│  低召回 (relevant@5≤1): {len(failures_1)}/{len(r.per_query)} ({len(failures_1) / max(len(r.per_query), 1) * 100:.1f}%)"
    )
    if failures_0:
        _add("│  零召回列表:")
        for f in failures_0[:10]:
            _add(f"│    - [{f['id']}] [{f['paper']}] [{f['type']}/{f['difficulty']}] {f['query'][:80]}...")
    _add("└─")

    # ── 生成指标 ──
    if g and g.per_query:
        _add()
        _add("┌─ 生成质量指标 (Generation Quality via LLM-as-Judge)")
        _add(f"│  评估样本数: {len(g.per_query)}")
        _add(f"│  Faithfulness (忠实度):      {g.faithfulness:.4f}")
        _add(f"│  Answer Relevance (回答相关): {g.answer_relevance:.4f}")
        _add(f"│  Context Relevance (上下文相关): {g.context_relevance:.4f}")
        _add(f"│  Hallucination Rate (幻觉率):   {g.hallucination_rate:.4f}")
        _add("└─")

    # ── 端到端指标 ──
    if e and e.per_query:
        _add()
        _add("┌─ 端到端回答质量指标 (End-to-End)")
        _add(f"│  ROUGE-L Precision:  {e.rouge_l_precision:.4f}")
        _add(f"│  ROUGE-L Recall:     {e.rouge_l_recall:.4f}")
        _add(f"│  ROUGE-L F1:         {e.rouge_l_f1:.4f}")
        _add(f"│  BLEU-1:             {e.bleu_1:.4f}")
        _add(f"│  BLEU-4:             {e.bleu_4:.4f}")
        _add(f"│  Keyword Coverage:   {e.exact_match_partial:.4f}")
        _add(f"│  Semantic Similarity:{e.semantic_similarity:.4f}")
        _add(f"│  Avg Answer Length:  {e.answer_length_avg:.0f} chars")
        _add("└─")

    # ── 综合评分 ──
    _add()
    _add("┌─ 综合评分 (Composite Score)")
    composite = _composite_score(result)
    _add(f"│  检索得分 (40%):    {composite['retrieval_score']:.4f}")
    _add(f"│  生成得分 (40%):    {composite['generation_score']:.4f}")
    _add(f"│  端到端得分 (20%):  {composite['e2e_score']:.4f}")
    _add("│  ──────────────────────────")
    _add(f"│  综合评分:           {composite['overall']:.4f} / 1.0000")
    _add(f"│  等级:               {composite['grade']}")
    _add("└─")
    _add()
    _add("=" * 72)

    report = "\n".join(lines)
    print(report)

    # ── 保存 JSON ──
    if save_json:
        json_path = Path(output_dir) / f"eval_{result.pipeline_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        json_data = {
            "pipeline_name": result.pipeline_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_stats": stats,
            "retrieval": {
                "recall_at_k": r.recall_at_k,
                "precision_at_k": r.precision_at_k,
                "ndcg_at_k": r.ndcg_at_k,
                "hit_at_k": r.hit_at_k,
                "mrr": r.mrr,
                "mrr_ci_95": list(r.mrr_ci),
                "map": r.map_score,
            },
            "generation": {
                "faithfulness": g.faithfulness if g else None,
                "answer_relevance": g.answer_relevance if g else None,
                "context_relevance": g.context_relevance if g else None,
                "hallucination_rate": g.hallucination_rate if g else None,
                "sample_size": len(g.per_query) if g else 0,
            }
            if g
            else None,
            "end_to_end": {
                "rouge_l_f1": e.rouge_l_f1,
                "bleu_1": e.bleu_1,
                "bleu_4": e.bleu_4,
                "keyword_coverage": e.exact_match_partial,
                "semantic_similarity": e.semantic_similarity,
                "avg_answer_length": e.answer_length_avg,
            },
            "composite": composite,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info("评估报告已保存: %s", json_path)

    return report


def _composite_score(result: FullEvalResult) -> dict:
    """计算综合评分。"""
    r = result.retrieval
    g = result.generation
    e = result.e2e

    # 检索得分：recall@5 + precision@5 + mrr + ndcg@5 平均，再平均到 0-1
    retrieval_score = (
        r.recall_at_k.get(5, 0) * 0.30
        + r.precision_at_k.get(5, 0) * 0.20
        + r.mrr * 0.20
        + r.ndcg_at_k.get(5, 0) * 0.20
        + r.map_score * 0.10
    )

    # 生成得分
    if g and g.per_query:
        generation_score = (
            g.faithfulness * 0.40
            + g.answer_relevance * 0.30
            + g.context_relevance * 0.20
            + (1 - g.hallucination_rate) * 0.10
        )
    else:
        generation_score = 0.0

    # 端到端得分
    e2e_score = e.rouge_l_f1 * 0.30 + e.semantic_similarity * 0.35 + e.exact_match_partial * 0.35

    overall = retrieval_score * 0.40 + generation_score * 0.40 + e2e_score * 0.20

    if overall >= 0.80:
        grade = "A (优秀)"
    elif overall >= 0.65:
        grade = "B (良好)"
    elif overall >= 0.50:
        grade = "C (一般)"
    elif overall >= 0.35:
        grade = "D (较差)"
    else:
        grade = "F (需要改进)"

    return {
        "retrieval_score": round(retrieval_score, 4),
        "generation_score": round(generation_score, 4),
        "e2e_score": round(e2e_score, 4),
        "overall": round(overall, 4),
        "grade": grade,
    }


def compare_pipelines(
    baseline_retrieve_fn,
    candidate_retrieve_fn,
    queries=None,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """对比 baseline vs candidate 两条检索管线。

    Returns:
        包含对比结果的 dict
    """
    if queries is None:
        queries = get_dataset()

    baseline = compute_retrieval_metrics(baseline_retrieve_fn, queries, k_values)
    candidate = compute_retrieval_metrics(candidate_retrieve_fn, queries, k_values)

    deltas = {}
    for metric_name in ["recall_at_k", "precision_at_k", "ndcg_at_k", "hit_at_k"]:
        for k in k_values:
            b = getattr(baseline, metric_name).get(k, 0)
            c = getattr(candidate, metric_name).get(k, 0)
            delta = c - b
            pct = f"{delta / b * 100:+.1f}%" if b > 0 else "N/A"
            deltas[f"{metric_name}@{k}"] = {
                "baseline": round(b, 4),
                "candidate": round(c, 4),
                "delta": round(delta, 4),
                "pct_change": pct,
            }

    for metric_name in ["mrr", "map_score"]:
        b = getattr(baseline, metric_name)
        c = getattr(candidate, metric_name)
        deltas[metric_name] = {
            "baseline": round(b, 4),
            "candidate": round(c, 4),
            "delta": round(c - b, 4),
        }

    return {"baseline": baseline, "candidate": candidate, "deltas": deltas}
