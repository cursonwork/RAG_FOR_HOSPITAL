"""检索管线评估框架（兼容层）。

核心指标逻辑统一在 src.evaluation.metrics，本模块提供向后兼容的 EvalCase 接口。
"""

from dataclasses import dataclass, field

from src.evaluation.metrics import _dcg, _is_relevant, _ndcg


@dataclass
class EvalCase:
    """单个评估用例。"""

    query: str
    label: str = ""
    relevant_phrases: list[str] = field(default_factory=list)
    relevant_sections: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """单次评估结果。"""

    name: str = ""
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)
    macro_avg: dict = field(default_factory=dict)


def _doc_is_relevant(doc, case: EvalCase) -> bool:
    """向后兼容适配器：接受 EvalCase 而非独立参数。"""
    return _is_relevant(doc, case.relevant_phrases, case.relevant_sections)


# 向后兼容别名
_compute_dcg = _dcg
_compute_ndcg = _ndcg


def _compute_mrr(relevances_per_query: list[list[int]]) -> float:
    """Mean Reciprocal Rank。"""
    rr_sum = 0.0
    for rels in relevances_per_query:
        for i, rel in enumerate(rels):
            if rel > 0:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(relevances_per_query) if relevances_per_query else 0.0


def run_eval(
    retrieve_fn,
    queries: list[EvalCase],
    k_values: tuple[int, ...] = (5, 10, 20),
    name: str = "pipeline",
) -> EvalResult:
    """评估单个检索管线。

    Args:
        retrieve_fn: callable(query) → list[Document]，接受查询字符串返回文档列表
        queries: 评估用例列表
        k_values: 评估的 top-k 值
        name: 管线名称

    Returns:
        EvalResult 包含各项指标
    """
    result = EvalResult(name=name)
    per_query_relevances: list[list[int]] = []
    k_max = max(k_values)

    for case in queries:
        try:
            docs = retrieve_fn(case.query)
        except Exception:
            docs = []

        # 对每个位置做相关/不相关标注
        rels = [1 if _doc_is_relevant(d, case) else 0 for d in docs[:k_max]]
        per_query_relevances.append(rels)

    # ── 逐 k 值计算 ──
    for k in k_values:
        recalls = []
        precisions = []
        ndcgs = []

        for i, _case in enumerate(queries):
            rels = per_query_relevances[i][:k]
            retrieved_relevant = sum(rels)
            # 分母至少为检索到的相关文档数（保守估计，recall ≤ 1.0）
            total_relevant = max(retrieved_relevant, 1)

            recalls.append(retrieved_relevant / total_relevant)
            precisions.append(retrieved_relevant / k if k > 0 else 0)
            ndcgs.append(_compute_ndcg(per_query_relevances[i], k))

        result.recall_at_k[k] = round(sum(recalls) / len(recalls), 4) if recalls else 0
        result.precision_at_k[k] = round(sum(precisions) / len(precisions), 4) if precisions else 0
        result.ndcg_at_k[k] = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0

    result.mrr = round(_compute_mrr(per_query_relevances), 4)

    result.per_query = [
        {
            "query": case.query,
            "label": case.label,
            "rels": per_query_relevances[i],
            "relevant@5": sum(per_query_relevances[i][:5]),
        }
        for i, case in enumerate(queries)
    ]

    # 宏平均总结
    result.macro_avg = {
        "recall@5": result.recall_at_k.get(5, 0),
        "precision@5": result.precision_at_k.get(5, 0),
        "ndcg@5": result.ndcg_at_k.get(5, 0),
        "mrr": result.mrr,
    }

    return result


def compare_pipelines(
    baseline_fn,
    candidate_fn,
    queries: list[EvalCase],
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """对比两条检索管线的效果。

    Returns:
        包含 baseline、candidate 结果及提升幅度的 dict
    """
    baseline = run_eval(baseline_fn, queries, k_values, name="baseline")
    candidate = run_eval(candidate_fn, queries, k_values, name="candidate")

    deltas = {}
    for metric in ["recall_at_k", "precision_at_k", "ndcg_at_k"]:
        for k in k_values:
            b = getattr(baseline, metric).get(k, 0)
            c = getattr(candidate, metric).get(k, 0)
            delta = c - b
            key = f"{metric}@{k}"
            deltas[key] = {
                "baseline": round(b, 4),
                "candidate": round(c, 4),
                "delta": round(delta, 4),
                "pct": f"{delta / b * 100:+.1f}%" if b > 0 else "N/A",
            }

    deltas["mrr"] = {
        "baseline": round(baseline.mrr, 4),
        "candidate": round(candidate.mrr, 4),
        "delta": round(candidate.mrr - baseline.mrr, 4),
    }

    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
    }


# ═══════════════════════════════════════════════════════════════
# Paper1 评估用例（20 条）
# ═══════════════════════════════════════════════════════════════


def load_paper1_queries() -> list[EvalCase]:
    """返回 paper1（结直肠癌深度学习预后预测）的 20 条评估用例。

    用例覆盖 4 个维度：精确术语匹配、语义理解、多步推理、方法学/技术。
    """
    return [
        # ── 精确术语匹配（5 条）──
        EvalCase(
            query="deep stroma score colorectal cancer prognosis",
            label="精确: deep stroma score",
            relevant_phrases=["deep stroma score", "stromal score", "stroma score based"],
        ),
        EvalCase(
            query="CMS4 colorectal cancer molecular subtype",
            label="精确: CMS4 subtype",
            relevant_phrases=["CMS4", "consensus molecular subtype"],
        ),
        EvalCase(
            query="VGG19 neural network histology images",
            label="精确: VGG19",
            relevant_phrases=["VGG19", "VGG-19"],
        ),
        EvalCase(
            query="TRIPOD statement prognostic prediction model",
            label="精确: TRIPOD",
            relevant_phrases=["TRIPOD", "transparent reporting"],
        ),
        EvalCase(
            query="CAF gene expression signature stromal fibroblasts",
            label="精确: CAF signature",
            relevant_phrases=["CAF", "cancer-associated fibroblast", "Isella"],
        ),
        # ── 语义理解（5 条）──
        EvalCase(
            query="How does the microenvironment affect colorectal cancer patient outcome?",
            label="语义: microenvironment",
            relevant_phrases=["stromal microenvironment", "tumor microenvironment", "stromal compartment"],
            relevant_sections=["Discussion", "Introduction"],
        ),
        EvalCase(
            query="What tissue classes can deep learning identify in colorectal histology?",
            label="语义: tissue classes",
            relevant_phrases=["nine tissue classes", "tissue classes", "adipose", "lymphocytes", "debris"],
            relevant_sections=["Training and testing"],
        ),
        EvalCase(
            query="How was the neural network training dataset created?",
            label="语义: training data",
            relevant_phrases=["NCT-HE-100K", "100,000 histological image", "training set"],
            relevant_sections=["Training and testing", "Patient cohorts"],
        ),
        EvalCase(
            query="What is the prognostic significance of stromal patterns in colorectal cancer?",
            label="语义: stromal prognosis",
            relevant_phrases=["prognostic", "stroma", "HR", "hazard ratio"],
            relevant_sections=["Results", "Deep stroma score"],
        ),
        EvalCase(
            query="How was the model validated on external patient cohorts?",
            label="语义: external validation",
            relevant_phrases=["validation cohort", "DACHS", "independent cohort", "generalizes"],
            relevant_sections=["Deep stroma score generalizes"],
        ),
        # ── 多步推理（5 条）──
        EvalCase(
            query="Compare the deep stroma score to the CAF gene expression signature as prognostic markers",
            label="推理: stroma vs CAF",
            relevant_phrases=["CAF", "deep stroma score", "gene expression", "compared"],
            relevant_sections=["Neural network assessment of the stromal"],
        ),
        EvalCase(
            query="How does tissue decomposition by CNN relate to consensus molecular subtypes?",
            label="推理: CNN vs CMS",
            relevant_phrases=["consensus molecular", "CMS", "tissue decomposition"],
            relevant_sections=["CNNs can decompose"],
        ),
        EvalCase(
            query="What statistical evidence supports deep stroma score as an independent prognostic factor?",
            label="推理: independent prognostic",
            relevant_phrases=["independent prognostic", "multivariate", "hazard ratio", "p ="],
            relevant_sections=["Neural network assessment", "Discussion"],
        ),
        EvalCase(
            query="How were optimal cutoffs for the stromal score determined?",
            label="推理: cutoff determination",
            relevant_phrases=["cutoff", "optimal", "Youden", "threshold"],
            relevant_sections=["Deep stroma score"],
        ),
        EvalCase(
            query="What are the limitations of using H&E histology for automated cancer prognosis?",
            label="推理: limitations",
            relevant_phrases=["limitation", "proof of concept", "retrospective", "validated prospectively"],
            relevant_sections=["Discussion"],
        ),
        # ── 方法学/技术（5 条）──
        EvalCase(
            query="What patient cohorts were used and what were their characteristics?",
            label="方法: patient cohorts",
            relevant_phrases=["NCT biobank", "DACHS", "TCGA", "UMM", "patient"],
            relevant_sections=["Patient cohorts"],
        ),
        EvalCase(
            query="How were neural networks trained and what data augmentation was used?",
            label="方法: training details",
            relevant_phrases=["rotational invariance", "data augmentation", "pretrained", "ImageNet"],
            relevant_sections=["Training and testing"],
        ),
        EvalCase(
            query="What statistical methods were used for survival analysis in this study?",
            label="方法: statistics",
            relevant_phrases=["Cox proportional hazards", "Kaplan-Meier", "log-rank", "multivariate"],
            relevant_sections=["Deep stroma score", "Neural network assessment"],
        ),
        EvalCase(
            query="How were whole-slide histological images preprocessed for deep learning?",
            label="方法: image preprocessing",
            relevant_phrases=["tiles", "patch", "whole-slide", "manually extracted"],
            relevant_sections=["Patient cohorts", "Training and testing"],
        ),
        EvalCase(
            query="What software and computational resources were used in this study?",
            label="方法: software tools",
            relevant_phrases=["R", "Python", "Nvidia", "GPU", "Keras", "TensorFlow"],
            relevant_sections=["Software"],
        ),
    ]
