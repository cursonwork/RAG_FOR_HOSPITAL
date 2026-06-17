"""
生成带分块标注的 PDF：在原文档上叠加彩色块，展示当前 RAG 分块管线效果。

用法:
    uv run python scripts/annotate_chunks.py                                    # paper1（默认）
    uv run python scripts/annotate_chunks.py --input data/documents/paper2.pdf  # 指定文件
    uv run python scripts/annotate_chunks.py --no-legend                        # 不加图例页
    uv run python scripts/annotate_chunks.py --format png                       # 输出 PNG 逐页
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import fitz
from langchain_core.documents import Document


def setup_project_path():
    """确保项目根目录在 sys.path 中，使 src.* 可导入。"""
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = setup_project_path()


def generate_colors(n: int) -> list[tuple[float, float, float]]:
    """生成 n 个视觉上分布均匀的 RGB 颜色（使用 golden ratio）。"""
    if n == 0:
        return []
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        # 饱和度 0.55，亮度 0.75 保证颜色不过于刺眼
        r, g, b = _hsl_to_rgb(hue, 0.55, 0.75)
        colors.append((r, g, b))
    return colors


def _hsl_to_rgb(h: float, s: float, lightness: float) -> tuple[float, float, float]:
    """HSL → RGB，返回值范围 [0, 1]。"""
    if s == 0:
        return (lightness, lightness, lightness)

    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = lightness * (1 + s) if lightness < 0.5 else lightness + s - lightness * s
    p = 2 * lightness - q
    return (hue_to_rgb(p, q, h + 1 / 3), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1 / 3))


def normalize_text(text: str) -> str:
    """规范化文本用于模糊匹配。"""
    # 折叠空白
    text = re.sub(r"\s+", " ", text)
    # 去 Markdown 标记
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # 去特殊字符噪声
    text = text.replace("​", "").replace("\xa0", " ")
    return text.strip()


def get_chunks(elements: list[dict], md_text: str, source: str, file_path: str) -> list[Document]:
    """用项目当前的分块管线处理文档（传入 ODL JSON 元素以启用 section-aware 分块）。"""
    from src.text_splitter import create_semantic_splitter

    splitter = create_semantic_splitter()
    doc = Document(
        page_content=md_text,
        metadata={
            "source": source,
            "file_path": file_path,
            "parser": "opendataloader",
            "odl_elements": elements,
        },
    )
    chunks = splitter([doc])
    return chunks


def match_elements_to_chunks(
    elements: list[dict],
    chunks: list[Document],
) -> tuple[dict, dict]:
    """将每个 JSON 元素匹配到所属的 chunk。

    优先使用 chunk metadata 中的 element_ids（精确匹配），
    回退到文本模糊匹配。

    返回 (element_id→chunk_idx, chunk_idx→[elements])
    """
    element_to_chunk: dict[int, int] = {}
    chunk_elements: dict[int, list[dict]] = defaultdict(list)
    element_map = {el["id"]: el for el in elements}

    # 优先：通过 element_ids 精确匹配
    has_element_ids = any("element_ids" in c.metadata for c in chunks)
    if has_element_ids:
        for i, chunk in enumerate(chunks):
            for eid in chunk.metadata.get("element_ids", []):
                element_to_chunk[eid] = i
                el = element_map.get(eid)
                if el:
                    chunk_elements[i].append(el)

        matched = len(element_to_chunk)
        total = sum(1 for el in elements if el.get("type") != "image" and (el.get("content") or "").strip())
        print(f"   精确匹配率 (element_ids): {matched}/{total}" if total else "   无文本元素")
        return element_to_chunk, chunk_elements

    # 回退：文本模糊匹配
    chunk_texts = [(i, normalize_text(c.page_content)) for i, c in enumerate(chunks)]

    for el in elements:
        content = (el.get("content") or "").strip()
        el_type = el.get("type", "")
        if not content or el_type == "image":
            continue

        el_norm = normalize_text(content)
        if len(el_norm) < 3:
            continue

        candidates: list[tuple[int, int]] = []  # (chunk_idx, match_len)

        for i, chunk_norm in chunk_texts:
            if not chunk_norm:
                continue

            # 精确子串匹配
            if el_norm in chunk_norm:
                candidates.append((i, len(el_norm)))
                continue

            # 长元素：取前 80 字符匹配
            if len(el_norm) > 80:
                head = el_norm[:80]
                if head in chunk_norm:
                    candidates.append((i, 80))
                    continue

            # 中等元素：取前 50 字符匹配
            if len(el_norm) > 50:
                head = el_norm[:50]
                if head in chunk_norm:
                    candidates.append((i, 50))
                    continue

            # 匹配 chunk 文本在元素内（超大元素可能包含整个 chunk）
            if len(chunk_norm) > 40 and chunk_norm[:60] in el_norm:
                candidates.append((i, 30))
                continue

            # 短元素：两方向都试试（允许部分匹配）
            if len(el_norm) >= 20:
                # 取元素的前 20 个字符逐个与 chunk 的前 20 个字符比较
                for overlap in range(20, min(len(el_norm), len(chunk_norm)), 20):
                    if el_norm[:overlap] in chunk_norm:
                        candidates.append((i, overlap))
                        break

        if candidates:
            # 选最长匹配
            best = max(candidates, key=lambda x: x[1])
            element_to_chunk[el["id"]] = best[0]
            chunk_elements[best[0]].append(el)

    return element_to_chunk, chunk_elements


def annotate_pdf(
    pdf_path: str,
    output_path: str,
    element_to_chunk: dict,
    chunk_elements: dict,
    chunks: list[Document],
    colors: list[tuple],
    add_legend: bool = True,
):
    """在 PDF 上叠加分块彩色标注。"""
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count

    page_elements_map: dict[int, list[dict]] = defaultdict(list)
    # 从 chunk_elements 反向构建页面元素映射
    for chunk_idx, els in chunk_elements.items():
        for el in els:
            page_num = el.get("page number", 1)
            page_elements_map[page_num].append((chunk_idx, el))

    # ---- 逐页绘制 ----
    for page_idx in range(num_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        items = page_elements_map.get(page_num, [])

        if not items:
            continue

        for chunk_idx, el in items:
            color = colors[chunk_idx % len(colors)]
            bbox = el.get("bounding box")
            if not bbox or len(bbox) != 4:
                continue

            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

            # 跳过异常大/小的框
            if rect.width <= 0 or rect.height <= 0:
                continue
            if rect.width > page.rect.width * 1.2 or rect.height > page.rect.height * 1.2:
                continue

            # 半透明填充块
            page.draw_rect(rect, color=color, fill=color, fill_opacity=0.22, width=0.3)

            # 左侧色条标记 chunk 编号
            left_bar = fitz.Rect(rect.x0 - 6, rect.y0, rect.x0 - 1, rect.y1)
            left_bar.x0 = max(left_bar.x0, 0)
            page.draw_rect(left_bar, color=color, fill=color, fill_opacity=0.85, width=0)

        # 页面上方的分块信息条
        _draw_page_header(doc, page_idx, page_num, items, colors, chunks)

    # ---- 图例页 ----
    if add_legend:
        _add_legend_page(doc, chunks, colors, chunk_elements)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"✅ 已保存: {output_path}")


def _draw_page_header(doc, page_idx: int, page_num: int, items, colors, chunks):
    """在页面顶部绘制该页涉及的分块编号条。"""
    page = doc[page_idx]

    # 找出本页唯一的 chunk_idx 列表（保持顺序）
    seen = set()
    page_chunks = []
    for chunk_idx, _ in items:
        if chunk_idx not in seen:
            seen.add(chunk_idx)
            page_chunks.append(chunk_idx)

    if not page_chunks:
        return

    # 不画色带，改用文字标注在右侧
    label = f"Chunks: {', '.join(str(c) for c in page_chunks)}"
    rect = fitz.Rect(page.rect.width - 250, 8, page.rect.width - 36, 22)
    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), fill_opacity=0.85, width=0)
    page.insert_textbox(
        rect,
        label,
        fontsize=8,
        color=(0.3, 0.3, 0.3),
        fontname="helv",
        align=2,
    )


def _add_legend_page(
    doc,
    chunks: list[Document],
    colors: list[tuple],
    chunk_elements: dict,
):
    """添加图例页，展示所有分块的详细信息。"""
    legend_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
    page_w = legend_page.rect.width
    margin = 50
    y = 40
    line_h = 16
    fontsize = 9

    # 标题
    legend_page.insert_text(
        fitz.Point(margin, y),
        f"RAG 分块分析 — 共 {len(chunks)} 个 Chunks",
        fontsize=14,
        color=(0.1, 0.1, 0.1),
        fontname="hebo",
    )
    y += 28

    # 分块配置信息
    from src.config import settings

    legend_page.insert_text(
        fitz.Point(margin, y),
        f"chunk_size={settings.chunk_size}  chunk_overlap={settings.chunk_overlap}  parser={settings.pdf_parser}",
        fontsize=9,
        color=(0.4, 0.4, 0.4),
        fontname="helv",
    )
    y += 22

    # 表头
    col_x = [margin, margin + 24, margin + 180, margin + 380]
    headers = ["#", "Section", "Chunk 内容预览"]
    legend_page.draw_line(fitz.Point(margin, y + 4), fitz.Point(page_w - margin, y + 4), color=(0.7, 0.7, 0.7))
    y += 8
    for col, header in zip(col_x, headers, strict=False):
        legend_page.insert_text(fitz.Point(col, y), header, fontsize=8, color=(0.4, 0.4, 0.4), fontname="helv")
    y += 10
    legend_page.draw_line(fitz.Point(margin, y), fitz.Point(page_w - margin, y), color=(0.7, 0.7, 0.7))
    y += 6

    for i, chunk in enumerate(chunks):
        if y > legend_page.rect.height - 50:
            legend_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
            y = 40

        color = colors[i % len(colors)]

        # 颜色方块
        color_rect = fitz.Rect(col_x[0], y, col_x[0] + 14, y + line_h)
        legend_page.draw_rect(color_rect, color=color, fill=color, fill_opacity=0.6, width=0.5)

        # Chunk 编号
        legend_page.insert_text(
            fitz.Point(col_x[0] + 18, y + line_h - 3),
            str(i),
            fontsize=fontsize - 1,
            color=(0.3, 0.3, 0.3),
            fontname="helv",
        )

        # Section 标题
        section = chunk.metadata.get("section_title", "")[:35]
        legend_page.insert_text(
            fitz.Point(col_x[1], y + line_h - 3),
            section if section else "(无标题)",
            fontsize=fontsize - 1,
            color=(0.2, 0.2, 0.2),
            fontname="helv",
        )

        # 内容预览
        preview = chunk.page_content[:100].replace("\n", " ")
        legend_page.insert_text(
            fitz.Point(col_x[2], y + line_h - 3),
            preview,
            fontsize=fontsize - 2,
            color=(0.3, 0.3, 0.3),
            fontname="helv",
        )

        # 该 chunk 覆盖的页数
        pages_in_chunk = sorted(set(el.get("page number", 1) for el in chunk_elements.get(i, [])))
        pages_str = f"pp.{min(pages_in_chunk)}-{max(pages_in_chunk)}" if pages_in_chunk else "—"
        legend_page.insert_text(
            fitz.Point(col_x[2] + 360, y + line_h - 3),
            pages_str,
            fontsize=fontsize - 2,
            color=(0.5, 0.5, 0.5),
            fontname="helv",
        )

        y += line_h + 1

    # 页脚说明
    legend_page.insert_text(
        fitz.Point(margin, legend_page.rect.height - 30),
        "每行 = 一个 RAG 分块 | 颜色对应正文标注中的色块 | 按分块顺序排列",
        fontsize=7,
        color=(0.5, 0.5, 0.5),
        fontname="helv",
    )


def main():
    parser = argparse.ArgumentParser(description="生成 RAG 分块标注 PDF")
    parser.add_argument("--input", default="data/documents/paper1_survival_prediction.pdf", help="输入 PDF 路径")
    parser.add_argument("--output", default=None, help="输出 PDF 路径（默认自动命名）")
    parser.add_argument("--no-legend", action="store_true", help="不生成图例页")
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf", help="输出格式")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        sys.exit(1)

    stem = input_path.stem
    output_path = args.output or f"data/{stem}_chunk_annotation.pdf"

    print(f"📄 处理: {input_path}")

    # ---- Step 1: opendataloader 提取 JSON + Markdown ----
    import opendataloader_pdf

    tmpdir = tempfile.mkdtemp(prefix="odl_annotate_")
    try:
        opendataloader_pdf.convert(
            input_path=str(input_path),
            output_dir=tmpdir,
            format=["json", "markdown"],
            quiet=True,
        )

        json_path = Path(tmpdir) / f"{stem}.json"
        md_path = Path(tmpdir) / f"{stem}.md"

        if not json_path.exists():
            print("❌ opendataloader 未生成 JSON，终止")
            sys.exit(1)
        if not md_path.exists():
            print("❌ opendataloader 未生成 Markdown，终止")
            sys.exit(1)

        j = json.loads(json_path.read_text(encoding="utf-8"))
        md_text = md_path.read_text(encoding="utf-8")

        elements = j.get("kids", [])
        text_elements = [el for el in elements if el.get("content", "").strip() and el.get("type") != "image"]
        print(f"   JSON 元素: {len(elements)} 个（其中文本元素 {len(text_elements)} 个）")
        print(f"   Markdown: {len(md_text)} 字符")

        # ---- Step 2: 运行分块管线 ----
        chunks = get_chunks(elements, md_text, input_path.name, str(input_path.absolute()))
        print(f"   分块数量: {len(chunks)}")
        for i, c in enumerate(chunks):
            sec = c.metadata.get("section_title", "") or "(无)"
            preview = c.page_content[:60].replace("\n", " ")
            print(f"      [{i}] section={sec} | {len(c.page_content)}chars | {preview}...")

        if not chunks:
            print("❌ 未生成任何分块，终止")
            sys.exit(1)

        # ---- Step 3: 匹配元素 → 分块 ----
        element_to_chunk, chunk_elements = match_elements_to_chunks(text_elements, chunks)
        matched = len(element_to_chunk)
        total = len(text_elements)
        print(f"   匹配率: {matched}/{total} ({100 * matched // total if total else 0}%)")

        # 未匹配元素 debug
        unmatched = [el for el in text_elements if el["id"] not in element_to_chunk]
        if unmatched and len(unmatched) <= 10:
            for el in unmatched:
                print(f'      ⚠ 未匹配: p{el.get("page number")} [{el.get("type")}] "{el["content"][:80]}"')

        # ---- Step 4: 生成颜色 ----
        colors = generate_colors(len(chunks))

        # ---- Step 5: 标注 PDF ----
        annotate_pdf(
            pdf_path=str(input_path),
            output_path=output_path,
            element_to_chunk=element_to_chunk,
            chunk_elements=chunk_elements,
            chunks=chunks,
            colors=colors,
            add_legend=not args.no_legend,
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
