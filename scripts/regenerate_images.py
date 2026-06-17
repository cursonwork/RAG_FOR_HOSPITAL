"""重新生成指定 PDF 的图片描述。

用法:
    uv run python scripts/regenerate_images.py
    uv run python scripts/regenerate_images.py --sources paper1 paper2  # 只处理指定论文
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from src.database import get_engine
from src.image_pipeline import fill_image_descriptions, save_image_placeholders

PDF_FILES = [
    "paper1_survival_prediction.pdf",
    "paper2_Automated_Risk_Classification_Colon_Biopsies.pdf",
    "paper3_MultiTask_SemiCOL_CRC_Histology.pdf",
    "paper4_CRC_Deep_Learning_Medical_Images_2025.pdf",
    "paper5_DL_MultiClass_Segmentation_CRC_SciRep2023.pdf",
    "paper6_CRC_AI_Narrative_Review_2025.pdf",
    "paper7_544.full.pdf",
]


def regenerate(source_patterns: list[str] | None = None) -> None:
    targets = [f for f in PDF_FILES if not source_patterns or any(p in f for p in source_patterns)]
    print(f"目标: {len(targets)} 篇论文")

    engine = get_engine()
    with engine.connect() as conn:
        for src in targets:
            # 删除该 source 的所有 images 记录
            result = conn.execute(
                text("DELETE FROM images WHERE source = :src RETURNING id"),
                {"src": src},
            )
            deleted = result.rowcount
            conn.commit()
            print(f"{'DEL' if deleted else 'SKIP'} {deleted:>4d} images — {src}")

    # 重新生成
    for src in targets:
        pdf_path = Path("data/documents") / src
        if not pdf_path.exists():
            print(f"SKIP (文件不存在): {pdf_path}")
            continue

        print(f"\n处理: {src}")
        records = save_image_placeholders(str(pdf_path), src)
        if records:
            fill_image_descriptions(records)
            print(f"  -> 完成 {len(records)} 张图片")
        else:
            print("  -> 无图片")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", help="部分 source 名称关键词")
    args = parser.parse_args()
    regenerate(args.sources)
