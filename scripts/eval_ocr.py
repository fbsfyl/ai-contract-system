"""文字版 vs 扫描版提取准确率对比（考核 E 的两套方案对比数据）。

方案一：文字版 PDF 直接用 pdfplumber 抽文本 → 提取。
方案二：扫描版 PDF 走 RapidOCR → 提取。
在同一份金标准（data/tests.json 的 test_purchase_1）上逐字段核对，量化 OCR 造成的准确率损失。

用法：python scripts/eval_ocr.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from app import extractor, ocr, pdf_loader  # noqa: E402
from evaluate import FIELDS, compare, field_equal  # noqa: E402

TESTS_PATH = os.path.join(BASE_DIR, "data", "tests.json")
FILLED_PDF = os.path.join(BASE_DIR, "data", "filled_contract.pdf")
SCANNED_PDF = os.path.join(BASE_DIR, "data", "scanned_contract.pdf")


def _load_gold() -> dict:
    with open(TESTS_PATH, encoding="utf-8") as f:
        tests = json.load(f)
    return next(x for x in tests if x["id"] == "test_purchase_1")["fields"]


def main() -> None:
    gold = _load_gold()

    # 方案一：文字版直接抽文本
    text_pdf, scene_pdf = pdf_loader.extract_text_from_pdf(FILLED_PDF)
    pred_text = extractor.extract(text_pdf, use_rag=False).model_dump()

    # 方案二：扫描件 OCR
    text_ocr = ocr.ocr_pdf(SCANNED_PDF)
    pred_ocr = extractor.extract(text_ocr, use_rag=False).model_dump()

    c_text, tot = compare(pred_text, gold)
    c_ocr, _ = compare(pred_ocr, gold)
    acc_text = c_text / tot * 100
    acc_ocr = c_ocr / tot * 100

    print("========== 文字版 vs 扫描版 准确率对比 ==========")
    print(f"文字版（pdfplumber 直抽）：{acc_text:.1f}%  ({c_text}/{tot})")
    print(f"扫描版（RapidOCR 识别）：{acc_ocr:.1f}%  ({c_ocr}/{tot})")
    print(f"差距：{acc_text - acc_ocr:+.1f}%")
    print("\n说明：两者差异反映 OCR 文本质量对下游提取的影响（零样本，排除 RAG 干扰）。")

    print("\n逐字段差异（仅列出两方案不一致的字段）：")
    diff = 0
    for k in FIELDS:
        t_ok = field_equal(k, pred_text.get(k), gold.get(k))
        o_ok = field_equal(k, pred_ocr.get(k), gold.get(k))
        if t_ok != o_ok:
            diff += 1
            print(f"  - {k}: 文字版={'对' if t_ok else '错'}, 扫描版={'对' if o_ok else '错'}, "
                  f"OCR值={pred_ocr.get(k)!r}")
    if diff == 0:
        print("  （两方案逐字段结果一致）")


if __name__ == "__main__":
    main()
