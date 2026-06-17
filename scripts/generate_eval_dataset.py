"""从生成的医学文档自动创建评估数据集 QA 对。

读取 data/md_documents/ 下的文档，用 DeepSeek 生成 EvalItem 格式的 QA 对，
输出 Python 代码片段可直接合并到 dataset.py。
"""

import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import create_llm
from src.logger import get_logger

logger = get_logger(__name__)

QA_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
你是一位医学RAG系统评估专家。请根据以下文档内容，生成3-5条检索评估用的QA对。

每条QA对必须包含：
1. question: 一个可以用文档内容回答的具体问题
2. reference_answer: 参考回答（基于文档内容的正确答案）
3. question_type: 问题类型（factoid/numerical/comparative/multi_hop/summary/terminology）
4. difficulty: 难度（easy/medium/hard）
5. relevant_phrases: 检索相关短语列表（文档中的关键词/术语）
6. must_contain_keywords: 回答必须包含的关键词

输出格式（严格JSON）：
```json
[
  {
    "question": "...",
    "reference_answer": "...",
    "question_type": "factoid",
    "difficulty": "easy",
    "relevant_phrases": ["...", "..."],
    "must_contain_keywords": ["...", "..."]
  }
]
```

要求：
- 问题覆盖不同类型和难度
- factoid: 事实类问题（定义、分类等）
- numerical: 数值类问题（剂量、发生率、指标值等）
- comparative: 比较类问题（对比不同方法/药物）
- multi_hop: 多跳推理（需要综合多处信息）
- summary: 总结类问题（概括主要内容）
- terminology: 术语类问题（解释专业术语）
- relevant_phrases 从文档原句中提取
- 回答要完整、准确，基于文档内容
- 只输出JSON，不要额外说明。\
""",
        ),
        (
            "user",
            """\
文档来源：{source}
文档类型：{doc_type}

文档内容：
{content}

请根据以上文档生成3-5条评估QA对。""",
        ),
    ]
)


def generate_qa_pairs(doc_path: str, doc_type: str) -> list[dict]:
    """为单个文档生成 QA 对。"""
    content = Path(doc_path).read_text(encoding="utf-8")
    source = Path(doc_path).name

    # 截取前 5000 字符（足够生成 QA，也减少 token 消耗）
    content_truncated = content[:5000]

    llm = create_llm(temperature=0.3)
    chain = QA_GENERATION_PROMPT | llm | StrOutputParser()

    try:
        response = chain.invoke(
            {
                "source": source,
                "doc_type": doc_type,
                "content": content_truncated,
            }
        )
    except Exception as e:
        logger.warning("QA生成失败 %s: %s", source, e)
        return []

    # 提取 JSON
    import json
    import re

    # 尝试提取 JSON 块
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    json_str = json_match.group(1) if json_match else response.strip()

    try:
        qa_items = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("JSON解析失败 %s: %s...", source, json_str[:200])
        return []

    # 添加 source 信息
    for item in qa_items:
        item["paper_source"] = source
        # 确保必填字段存在
        item.setdefault("relevant_phrases", item.get("must_contain_keywords", [])[:5])
        item.setdefault("must_contain_keywords", [])
        item.setdefault("difficulty", "medium")
        item.setdefault("question_type", "factoid")

    return qa_items


def generate_eval_dataset(
    base_dir: str = "data/md_documents",
    sample_per_type: int = 3,
) -> str:
    """为每类文档采样并生成评估 QA 对，输出 Python 代码。

    Args:
        base_dir: MD 文档目录
        sample_per_type: 每类文档采样数量

    Returns:
        可直接粘贴到 dataset.py 的 Python 代码字符串
    """
    base = Path(base_dir)
    types = {
        "consultations": "问诊记录",
        "textbook": "教材章节",
        "papers": "研究论文",
        "drug_manual": "药品手册",
        "symposium": "座谈报告",
        "cases": "病例报告",
    }

    all_qa_items: list[dict] = []
    prefix_map = {
        "consultations": "cons",
        "textbook": "text",
        "papers": "paper",
        "drug_manual": "drug",
        "symposium": "symp",
        "cases": "case",
    }

    for dir_name, doc_type in types.items():
        doc_dir = base / dir_name
        if not doc_dir.exists():
            continue

        md_files = sorted(doc_dir.glob("*.md"))
        if not md_files:
            continue

        # 采样（均匀间隔）
        step = max(1, len(md_files) // sample_per_type)
        sampled = md_files[::step][:sample_per_type]

        logger.info("处理 %s: %d 个文件 → 采样 %d 个", doc_type, len(md_files), len(sampled))

        for md_path in sampled:
            qa_items = generate_qa_pairs(str(md_path), doc_type)
            for item in qa_items:
                item["_prefix"] = prefix_map[dir_name]
                item["_doc_type"] = doc_type
            all_qa_items.extend(qa_items)
            logger.info("  %s → %d 条QA", md_path.name, len(qa_items))

    # 生成 Python 代码
    lines = []
    lines.append(f"# 自动生成的评估数据集 — {len(all_qa_items)} 条 QA 对")
    lines.append(f"# 来源：{base_dir}/ 下的 6 类医学文档")
    lines.append("")

    counter = {}
    for item in all_qa_items:
        prefix = item["_prefix"]
        counter[prefix] = counter.get(prefix, 0) + 1
        qid = f"{prefix}_q{counter[prefix]:02d}"

        escaped_answer = item["reference_answer"].replace('"', '\\"').replace("\n", "\\n")

        lines.append("EvalItem(")
        lines.append(f'    id="{qid}",')
        lines.append(f'    question="{item["question"]}",')
        lines.append(f'    reference_answer="{escaped_answer}",')
        lines.append(f'    paper_source="{item["paper_source"]}",')
        lines.append(f'    question_type="{item["question_type"]}",')
        lines.append(f'    difficulty="{item["difficulty"]}",')
        lines.append(f"    relevant_phrases={item['relevant_phrases']},")
        lines.append(f"    must_contain_keywords={item['must_contain_keywords']},")
        lines.append("),")
        lines.append("")

    code = "\n".join(lines)

    # 保存到文件
    output_path = base / ".." / "generated_qa_items.py"
    Path(output_path).write_text(code, encoding="utf-8")
    logger.info("已生成 %d 条 QA 对 → %s", len(all_qa_items), output_path)

    return code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从医学文档自动生成评估QA对")
    parser.add_argument("--sample", type=int, default=3, help="每类采样数")
    parser.add_argument("--output", default="data/generated_qa_items.py", help="输出文件")
    args = parser.parse_args()

    code = generate_eval_dataset(sample_per_type=args.sample)
    print(code[:500])  # 预览
    print(f"\n... (共 {len(code)} 字符)")
