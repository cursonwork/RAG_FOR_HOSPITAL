"""PDF 图片管线：提取 → 多模态 LLM 描述 → 向量化存储。

第一阶段（save_image_placeholders）：提取图片 → 写 PG + Milvus 占位（描述为空）。
第二阶段（fill_image_descriptions）：并发调用 qwen → UPDATE PG 已有记录。
"""

import base64
import io
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fitz
from openai import OpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

IMAGE_DESCRIPTION_PROMPT = """Describe this medical image concisely.

Reply in the same language as the image caption. If no caption, reply in Chinese.

Include:
1. Image type (chart / radiology / pathology slide / diagram / photo / table)
2. Key visual elements and any data shown
3. Medical/clinical significance if apparent

Image caption: {caption}

Output only the description, no prefixes like "This image shows..." or "这张图片显示了..."."""



@dataclass
class ImageRecord:
    id: str
    image_data: bytes
    image_format: str
    description: str
    caption: str
    source: str
    page: int
    bbox: tuple | None
    chunk_id: str = ""


def _get_qwen_client() -> OpenAI:
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )


def _compress_image(image_bytes: bytes, max_size: int = 800) -> bytes:
    """压缩图片到指定最大边长，返回 JPEG 编码。"""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def describe_image(image_bytes: bytes, caption: str = "", client: OpenAI | None = None) -> str:
    """用 qwen 多模态模型描述图片内容（单次调用）。"""
    if client is None:
        client = _get_qwen_client()

    try:
        compressed = _compress_image(image_bytes, settings.image_max_size)
        img_b64 = base64.b64encode(compressed).decode("utf-8")
    except Exception:
        logger.exception("图片压缩/编码失败")
        return ""

    prompt = IMAGE_DESCRIPTION_PROMPT.format(caption=caption or "无标题")

    try:
        response = client.chat.completions.create(
            model=settings.dashscope_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
        )
        description = response.choices[0].message.content or ""
        return description.strip()
    except Exception:
        logger.exception("qwen 多模态 API 调用失败")
        return ""


def _find_caption_for_image(page: fitz.Page, img_bbox: tuple, caption_candidates: list[dict]) -> str:
    """在同页 JSON 元素中找离图片最近的 caption/paragraph。"""
    if not img_bbox:
        return ""
    x0, y0, x1, y1 = img_bbox
    img_bottom = y1
    img_center_x = (x0 + x1) / 2

    best: str = ""
    best_dist = float("inf")

    for elem in caption_candidates:
        bbox = elem.get("bounding box", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        ex0, ey0, ex1, ey1 = bbox

        if ey0 >= img_bottom:
            dist = ey0 - img_bottom
            h_center = (ex0 + ex1) / 2
            h_diff = abs(h_center - img_center_x)
            if dist < best_dist and h_diff < (x1 - x0) * 1.5:
                best_dist = dist
                best = elem.get("content", "")

    return best


def extract_images_from_pdf(file_path: str, json_data: dict | None = None) -> list[dict]:
    """从 PDF 逐页提取嵌入图片，返回图片信息列表。

    每项: {image_bytes, format, page, caption, bbox}
    """
    doc = fitz.open(file_path)
    images: list[dict] = []

    caption_map: dict[int, list[dict]] = {}
    if json_data:
        for kid in json_data.get("kids", []):
            pn = kid.get("page number", 0)
            t = kid.get("type", "")
            if t in ("caption", "paragraph"):
                caption_map.setdefault(pn, []).append(kid)

    for page_num, page in enumerate(doc, start=1):
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image.get("image")
            if not image_bytes:
                continue

            # 过滤小图（icon / logo 等噪声）
            if len(image_bytes) < settings.image_min_bytes:
                continue

            fmt = base_image.get("ext", "png")
            bbox = page.get_image_bbox(img_info)

            # raw bbox 可能是 Rect 对象，转为 tuple
            if bbox is not None:
                bbox = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)

            caption = ""
            if json_data and page_num in caption_map and bbox:
                caption = _find_caption_for_image(page, bbox, caption_map[page_num])
            if not caption:
                titles = []
                for t in page.get_text("blocks"):
                    if t[6] == 0:
                        txt = t[4].strip()
                        if txt and len(txt) < 200:
                            titles.append(txt)
                if titles:
                    caption = titles[-1]

            images.append(
                {
                    "image_bytes": image_bytes,
                    "format": fmt,
                    "page": page_num,
                    "caption": caption,
                    "bbox": bbox,
                }
            )

    doc.close()
    return images


# ── 第一阶段：提取图片并写入占位记录 ──


def save_image_placeholders(file_path: str, source_name: str) -> list[ImageRecord]:
    """提取图片 → 写 PG + 写 Milvus 占位（描述留空）。返回 ImageRecord 列表供第二阶段。"""
    from contextlib import suppress

    json_data = None
    if settings.pdf_parser == "opendataloader":
        stem = Path(file_path).stem
        json_path = Path(file_path).parent / f"{stem}.json"
        if json_path.exists():
            with suppress(Exception):
                json_data = json.loads(json_path.read_text(encoding="utf-8"))

    images = extract_images_from_pdf(file_path, json_data)
    if not images:
        logger.info("%s: 未找到嵌入图片", source_name)
        return []

    logger.info("%s: 发现 %d 张图片，写入占位...", source_name, len(images))
    records: list[ImageRecord] = []

    for img in images:
        img_id = uuid.uuid4().hex[:16]
        record = ImageRecord(
            id=img_id,
            image_data=img["image_bytes"],
            image_format=img["format"],
            description="",  # 第二阶段填充
            caption=img["caption"],
            source=source_name,
            page=img["page"],
            bbox=img["bbox"],
        )
        records.append(record)

    # 写 Milvus（用 caption 做向量，描述为空）
    from src.vector_store import get_vector_store

    store = get_vector_store()
    store.add_image_placeholders(records)

    logger.info("%s: 占位写入完成，%d 张图片", source_name, len(records))
    return records


# ── 第二阶段：并发生成描述并 UPDATE ──


def _describe_one(record: ImageRecord, client: OpenAI) -> ImageRecord:
    """并发任务：描述单张图片。"""
    desc = describe_image(record.image_data, record.caption, client)
    record.description = desc
    return record


def fill_image_descriptions(records: list[ImageRecord]) -> list[ImageRecord]:
    """并发生成描述 → 批量 UPDATE PG + 重建 Milvus 向量。"""
    if not records:
        return []

    client = _get_qwen_client()
    total = len(records)
    logger.info("开始并发生成 %d 张图片描述 (max_workers=%d)...", total, settings.image_max_concurrent)

    completed: list[ImageRecord] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=settings.image_max_concurrent) as executor:
        futures = {executor.submit(_describe_one, r, client): i for i, r in enumerate(records)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                record = future.result()
                if record.description:
                    completed.append(record)
                else:
                    failed += 1
            except Exception:
                logger.exception("图片 %d 并发处理异常", idx + 1)
                failed += 1
            if (len(completed) + failed) % 10 == 0:
                logger.info("图片描述进度: %d/%d", len(completed) + failed, total)

    logger.info("图片描述完成: %d 成功, %d 失败, %d 总计", len(completed), failed, total)

    # 批量 UPDATE PG + 重建 Milvus
    if completed:
        from src.database import update_image_description

        for r in completed:
            update_image_description(r.id, r.description)
        # 重建向量（用新描述）
        from src.vector_store import get_vector_store

        get_vector_store().update_image_vectors(completed)
        logger.info("图片向量已更新: %d 张", len(completed))

    return completed
