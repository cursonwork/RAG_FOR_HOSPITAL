"""RAG 系统评估指标模块。

包含三类指标：
1. 检索指标：Recall@k, Precision@k, NDCG@k, MRR, MAP, Hit@k
2. 生成指标：Faithfulness, Answer Relevance, Context Relevance（LLM-as-judge）
3. 端到端指标：ROUGE-L, BLEU-1/4, Semantic Similarity, Token Efficiency
"""

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_core.documents import Document


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    """检索指标结果。"""
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    hit_at_k: dict[int, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)

    # 置信区间（95% bootstrap）
    recall_at_k_ci: dict[int, tuple[float, float]] = field(default_factory=dict)
    mrr_ci: tuple[float, float] = (0, 0)


@dataclass
class GenerationMetrics:
    """生成质量指标结果（LLM-as-judge）。"""
    faithfulness: float = 0.0  # 生成回答对检索上下文的忠实度
    answer_relevance: float = 0.0  # 回答与问题的相关性
    context_relevance: float = 0.0  # 检索上下文与问题的相关性
    hallucination_rate: float = 0.0  # 幻觉率
    per_query: list[dict] = field(default_factory=list)


@dataclass
class EndToEndMetrics:
    """端到端回答质量指标。"""
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    rouge_l_f1: float = 0.0
    bleu_1: float = 0.0
    bleu_4: float = 0.0
    semantic_similarity: float = 0.0  # 基于 embedding 的语义相似度
    exact_match_partial: float = 0.0  # 部分匹配率（关键词覆盖率）
    answer_length_avg: float = 0.0
    per_query: list[dict] = field(default_factory=list)


@dataclass
class FullEvalResult:
    """完整评估结果。"""
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics | None = None  # None 表示未运行 LLM 评估
    e2e: EndToEndMetrics = field(default_factory=EndToEndMetrics)
    dataset_stats: dict = field(default_factory=dict)
    pipeline_name: str = ""


# ═══════════════════════════════════════════════════════════════
# 1. 检索指标
# ═══════════════════════════════════════════════════════════════

def _is_relevant(doc: Document, relevant_phrases: list[str],
                 relevant_sections: list[str]) -> bool:
    """判定单个文档是否与查询相关。"""
    content = doc.page_content.lower()
    for phrase in relevant_phrases:
        if phrase.lower() in content:
            return True
    section = (doc.metadata.get("section") or "").lower()
    for sec in relevant_sections:
        if sec.lower() in section:
            return True
    return False


def _dcg(rels: list[int], k: int) -> float:
    d = 0.0
    for i, rel in enumerate(rels[:k]):
        d += rel / math.log2(i + 2)
    return d


def _ndcg(rels: list[int], k: int) -> float:
    d = _dcg(rels, k)
    ideal = sorted(rels, reverse=True)
    idc = _dcg(ideal, k)
    return d / idc if idc > 0 else 0.0


def _average_precision(rels: list[int]) -> float:
    """计算 Average Precision（用于 MAP）。"""
    num_relevant = sum(rels)
    if num_relevant == 0:
        return 0.0
    ap = 0.0
    relevant_count = 0
    for i, rel in enumerate(rels):
        if rel == 1:
            relevant_count += 1
            ap += relevant_count / (i + 1)
    return ap / num_relevant


def _bootstrap_ci(data: list[float], n_bootstrap: int = 2000,
                  ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% 置信区间（百分位法）。"""
    import random
    n = len(data)
    if n < 3:
        return (0, 0)
    means = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lower = means[int((1 - ci) / 2 * n_bootstrap)]
    upper = means[int((1 + ci) / 2 * n_bootstrap)]
    return (round(lower, 4), round(upper, 4))


def compute_retrieval_metrics(
    retrieve_fn,
    queries: list,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> RetrievalMetrics:
    """计算所有检索指标。

    Args:
        retrieve_fn: callable(query_str) → list[Document]
        queries: EvalItem 列表
        k_values: 评估的 top-k 值

    Returns:
        RetrievalMetrics 包含所有指标
    """
    result = RetrievalMetrics()
    k_max = max(k_values)
    all_rels: list[list[int]] = []
    all_precisions_for_map: list[float] = []

    for item in queries:
        try:
            docs = retrieve_fn(item.question)
        except Exception:
            docs = []

        rels = [
            1 if _is_relevant(d, item.relevant_phrases, item.relevant_sections)
            else 0
            for d in docs[:k_max]
        ]
        all_rels.append(rels)
        all_precisions_for_map.append(_average_precision(rels))

        result.per_query.append({
            "id": item.id,
            "query": item.question[:100],
            "paper": item.paper_source,
            "type": item.question_type,
            "difficulty": item.difficulty,
            "rels": rels,
            "relevant@5": sum(rels[:5]),
            "relevant@10": sum(rels[:10]),
        })

    # ── Per-k metrics ──
    for k in k_values:
        recalls, precisions, ndcgs, hits = [], [], [], []

        for i, item in enumerate(queries):
            rels_k = all_rels[i][:k]
            retrieved_relevant = sum(rels_k)
            total_relevant = max(retrieved_relevant, 1)

            recalls.append(retrieved_relevant / total_relevant)
            precisions.append(retrieved_relevant / k if k > 0 else 0)
            ndcgs.append(_ndcg(all_rels[i], k))
            hits.append(1.0 if retrieved_relevant > 0 else 0.0)

        result.recall_at_k[k] = round(statistics.mean(recalls), 4) if recalls else 0
        result.precision_at_k[k] = round(statistics.mean(precisions), 4) if precisions else 0
        result.ndcg_at_k[k] = round(statistics.mean(ndcgs), 4) if ndcgs else 0
        result.hit_at_k[k] = round(statistics.mean(hits), 4) if hits else 0

        # Bootstrap CI for recall
        result.recall_at_k_ci[k] = _bootstrap_ci(recalls)

    # ── MRR ──
    rr_list = []
    for rels in all_rels:
        for i, rel in enumerate(rels):
            if rel > 0:
                rr_list.append(1.0 / (i + 1))
                break
        else:
            rr_list.append(0.0)
    result.mrr = round(statistics.mean(rr_list), 4) if rr_list else 0
    result.mrr_ci = _bootstrap_ci(rr_list)

    # ── MAP ──
    result.map_score = round(statistics.mean(all_precisions_for_map), 4) if all_precisions_for_map else 0

    return result


# ═══════════════════════════════════════════════════════════════
# 2. 生成指标（LLM-as-judge）
# ═══════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """你是一个评估RAG系统输出质量的专家。请评估以下"生成回答"对"检索上下文"的忠实度（Faithfulness）。

忠实度定义：回答中的所有事实性陈述是否都可以从给定的上下文中推断出来。
- 分数 1.0: 所有陈述都直接来自上下文，没有编造
- 分数 0.7-0.9: 大部分来自上下文，有少量不重要的推测
- 分数 0.4-0.6: 约一半来自上下文，有若干无法验证的陈述
- 分数 0.0-0.3: 大量编造或与上下文矛盾

问题：{question}

检索到的上下文：
{context}

生成回答：
{answer}

请给出 0.0 到 1.0 之间的分数，并简要说明理由。
格式：SCORE: <数字>
REASON: <一句话理由>"""

ANSWER_RELEVANCE_PROMPT = """你是一个评估RAG系统输出质量的专家。请评估以下"生成回答"对"问题"的相关性（Answer Relevance）。

相关性定义：回答是否直接、完整地回应了问题，没有偏离主题。
- 分数 1.0: 完全切题，完整回答了问题的所有部分
- 分数 0.7-0.9: 基本切题，但遗漏了次要方面
- 分数 0.4-0.6: 部分切题，主要方面被遗漏或偏题
- 分数 0.0-0.3: 基本不切题，没有回答所问的问题

问题：{question}

生成回答：
{answer}

请给出 0.0 到 1.0 之间的分数，并简要说明理由。
格式：SCORE: <数字>
REASON: <一句话理由>"""

CONTEXT_RELEVANCE_PROMPT = """你是一个评估RAG系统检索质量的专家。请评估以下"检索到的上下文"与"问题"的相关性（Context Relevance）。

上下文相关性定义：检索到的文档是否与问题相关，是否包含回答问题所需的信息。
- 分数 1.0: 所有检索到的文档都相关，包含回答所需的完整信息
- 分数 0.7-0.9: 大部分文档相关，包含大部分所需信息
- 分数 0.4-0.6: 约一半文档相关，信息不完整
- 分数 0.0-0.3: 大部分文档不相关，缺乏关键信息

问题：{question}

检索到的上下文：
{context}

请给出 0.0 到 1.0 之间的分数，并简要说明理由。
格式：SCORE: <数字>
REASON: <一句话理由>"""


def _parse_llm_score(response: str) -> tuple[float, str]:
    """解析 LLM 评估输出中的分数。"""
    score = 0.0
    reason = ""
    for line in response.strip().split("\n"):
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return max(0.0, min(1.0, score)), reason


def compute_generation_metrics(
    llm,
    queries: list,
    answers: list[str],
    contexts: list[str],
    sample_size: int = 20,
) -> GenerationMetrics:
    """使用 LLM-as-judge 计算生成质量指标。

    Args:
        llm: LangChain ChatOpenAI 实例
        queries: EvalItem 列表
        answers: 生成的回答列表（与 queries 一一对应）
        contexts: 检索到的上下文字符串列表（与 queries 一一对应）
        sample_size: LLM 评估的采样数量（避免过多 API 调用）

    Returns:
        GenerationMetrics
    """
    result = GenerationMetrics()

    # 分层采样
    if len(queries) > sample_size:
        step = max(1, len(queries) // sample_size)
        indices = list(range(0, len(queries), step))[:sample_size]
    else:
        indices = list(range(len(queries)))

    f_scores, ar_scores, cr_scores = [], [], []

    for idx in indices:
        item = queries[idx]
        answer = answers[idx] if idx < len(answers) else ""
        context = contexts[idx] if idx < len(contexts) else ""

        if not answer or not context:
            continue

        # Faithfulness
        prompt_f = FAITHFULNESS_PROMPT.format(
            question=item.question, context=context[:4000], answer=answer[:2000]
        )
        try:
            resp_f = llm.invoke(prompt_f)
            score_f, reason_f = _parse_llm_score(resp_f.content)
        except Exception:
            score_f, reason_f = 0.0, "eval failed"

        # Answer Relevance
        prompt_ar = ANSWER_RELEVANCE_PROMPT.format(
            question=item.question, answer=answer[:2000]
        )
        try:
            resp_ar = llm.invoke(prompt_ar)
            score_ar, reason_ar = _parse_llm_score(resp_ar.content)
        except Exception:
            score_ar, reason_ar = 0.0, "eval failed"

        # Context Relevance
        prompt_cr = CONTEXT_RELEVANCE_PROMPT.format(
            question=item.question, context=context[:4000]
        )
        try:
            resp_cr = llm.invoke(prompt_cr)
            score_cr, reason_cr = _parse_llm_score(resp_cr.content)
        except Exception:
            score_cr, reason_cr = 0.0, "eval failed"

        f_scores.append(score_f)
        ar_scores.append(score_ar)
        cr_scores.append(score_cr)

        result.per_query.append({
            "id": item.id,
            "faithfulness": score_f,
            "answer_relevance": score_ar,
            "context_relevance": score_cr,
            "faithfulness_reason": reason_f,
            "ar_reason": reason_ar,
            "cr_reason": reason_cr,
        })

    if f_scores:
        result.faithfulness = round(statistics.mean(f_scores), 4)
        result.answer_relevance = round(statistics.mean(ar_scores), 4)
        result.context_relevance = round(statistics.mean(cr_scores), 4)
        # 幻觉率 = 与上下文不符的陈述（faithfulness < 0.5 的比例）
        result.hallucination_rate = round(
            sum(1 for s in f_scores if s < 0.5) / len(f_scores), 4
        )

    return result


# ═══════════════════════════════════════════════════════════════
# 3. 端到端指标
# ═══════════════════════════════════════════════════════════════

def _ngram_overlap(text1: str, text2: str, n: int) -> tuple[int, int, int]:
    """计算 n-gram 重叠。返回 (overlap, total_pred, total_ref)。"""
    def ngrams(text, n):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    ng1 = ngrams(text1, n)
    ng2 = ngrams(text2, n)
    overlap = len(ng1 & ng2)
    return overlap, len(ng1), len(ng2)


def _rouge_l(pred: str, ref: str) -> tuple[float, float, float]:
    """计算 ROUGE-L（最长公共子序列）。"""
    import difflib
    s = difflib.SequenceMatcher(None, pred.lower().split(), ref.lower().split())
    lcs = sum(block.size for block in s.get_matching_blocks())
    precision = lcs / max(len(pred.split()), 1)
    recall = lcs / max(len(ref.split()), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _bleu_score(pred: str, ref: str, max_n: int = 4) -> dict[int, float]:
    """计算 BLEU-n 分数（简化版，不含 brevity penalty）。"""
    scores = {}
    for n in range(1, max_n + 1):
        overlap, total_pred, _ = _ngram_overlap(pred, ref, n)
        scores[n] = overlap / max(total_pred, 1)
    return scores


def _keyword_coverage(pred: str, keywords: list[str]) -> float:
    """关键词覆盖率。"""
    if not keywords:
        return 0.0
    pred_lower = pred.lower()
    covered = sum(1 for kw in keywords if kw.lower() in pred_lower)
    return covered / len(keywords)


def _semantic_similarity(embeddings, pred: str, ref: str) -> float:
    """基于 embedding 的语义相似度（余弦相似度）。"""
    try:
        pred_emb = embeddings.embed_query(pred)
        ref_emb = embeddings.embed_query(ref)
        dot = sum(a * b for a, b in zip(pred_emb, ref_emb))
        norm_p = math.sqrt(sum(a * a for a in pred_emb))
        norm_r = math.sqrt(sum(b * b for b in ref_emb))
        if norm_p == 0 or norm_r == 0:
            return 0.0
        return dot / (norm_p * norm_r)
    except Exception:
        return 0.0


def compute_e2e_metrics(
    queries: list,
    answers: list[str],
    embeddings=None,
) -> EndToEndMetrics:
    """计算端到端回答质量指标。

    Args:
        queries: EvalItem 列表
        answers: 生成的回答列表
        embeddings: create_embeddings() 实例（用于语义相似度）

    Returns:
        EndToEndMetrics
    """
    result = EndToEndMetrics()
    rl_p, rl_r, rl_f = [], [], []
    b1_list, b4_list = [], []
    em_list = []
    sem_list = []
    lengths = []

    for item, answer in zip(queries, answers):
        ref = item.reference_answer
        if not answer or not ref:
            continue

        # ROUGE-L
        p, r, f = _rouge_l(answer, ref)
        rl_p.append(p)
        rl_r.append(r)
        rl_f.append(f)

        # BLEU
        bleu_scores = _bleu_score(answer, ref)
        b1_list.append(bleu_scores[1])
        b4_list.append(bleu_scores[4])

        # Keyword coverage
        if item.must_contain_keywords:
            em_list.append(_keyword_coverage(answer, item.must_contain_keywords))

        # Semantic similarity
        if embeddings:
            sem_list.append(_semantic_similarity(embeddings, answer, ref))

        lengths.append(len(answer))

        result.per_query.append({
            "id": item.id,
            "rouge_l_f1": round(f, 4),
            "bleu_1": round(bleu_scores[1], 4),
            "keyword_coverage": round(em_list[-1], 4) if em_list else 0,
        })

    if rl_f:
        result.rouge_l_precision = round(statistics.mean(rl_p), 4)
        result.rouge_l_recall = round(statistics.mean(rl_r), 4)
        result.rouge_l_f1 = round(statistics.mean(rl_f), 4)
    if b1_list:
        result.bleu_1 = round(statistics.mean(b1_list), 4)
        result.bleu_4 = round(statistics.mean(b4_list), 4)
    if em_list:
        result.exact_match_partial = round(statistics.mean(em_list), 4)
    if sem_list:
        result.semantic_similarity = round(statistics.mean(sem_list), 4)
    if lengths:
        result.answer_length_avg = round(statistics.mean(lengths), 1)

    return result


# ═══════════════════════════════════════════════════════════════
# 4. 切片分析
# ═══════════════════════════════════════════════════════════════

def compute_slice_metrics(
    retrieve_fn,
    queries: list,
    k_value: int = 5,
) -> dict[str, dict]:
    """按维度对检索结果进行切片分析。

    Returns:
        {slice_name: {"count": n, "recall@5": x, "precision@5": y, "mrr": z}}
    """
    slices = {}

    # 按 paper 切片
    by_paper = defaultdict(list)
    for item in queries:
        by_paper[item.paper_source].append(item)

    for paper, items in by_paper.items():
        metrics = compute_retrieval_metrics(retrieve_fn, items, k_values=(k_value,))
        short = paper.replace(".pdf", "").replace("paper", "P")
        slices[f"paper:{short}"] = {
            "count": len(items),
            f"recall@{k_value}": metrics.recall_at_k[k_value],
            f"precision@{k_value}": metrics.precision_at_k[k_value],
            f"ndcg@{k_value}": metrics.ndcg_at_k[k_value],
            "mrr": metrics.mrr,
        }

    # 按问题类型切片
    by_type = defaultdict(list)
    for item in queries:
        by_type[item.question_type].append(item)

    for qtype, items in by_type.items():
        if len(items) < 2:
            continue
        metrics = compute_retrieval_metrics(retrieve_fn, items, k_values=(k_value,))
        slices[f"type:{qtype}"] = {
            "count": len(items),
            f"recall@{k_value}": metrics.recall_at_k[k_value],
            f"precision@{k_value}": metrics.precision_at_k[k_value],
            f"ndcg@{k_value}": metrics.ndcg_at_k[k_value],
            "mrr": metrics.mrr,
        }

    # 按难度切片
    by_difficulty = defaultdict(list)
    for item in queries:
        by_difficulty[item.difficulty].append(item)

    for diff, items in by_difficulty.items():
        metrics = compute_retrieval_metrics(retrieve_fn, items, k_values=(k_value,))
        slices[f"difficulty:{diff}"] = {
            "count": len(items),
            f"recall@{k_value}": metrics.recall_at_k[k_value],
            f"precision@{k_value}": metrics.precision_at_k[k_value],
            f"ndcg@{k_value}": metrics.ndcg_at_k[k_value],
            "mrr": metrics.mrr,
        }

    return dict(slices)
