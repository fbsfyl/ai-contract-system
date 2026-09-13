"""验证复杂表格管线（PP-Structure 最小集）。

流程：生成含中文表格的 PDF → 渲染成图片 → 倾斜/预处理 → 表格结构识别 → 单元格 OCR。
用法：python scripts/test_table.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from app import table as table_pipeline

OUT = os.path.join(BASE_DIR, "data", "table_contract.pdf")
FONT = "STSong-Light"


def _generate_table_pdf(path: str):
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    doc = SimpleDocTemplate(path, pagesize=A4)
    data = [
        ["付款节点", "金额（元）", "比例", "支付条件"],
        ["预付款", "120000", "30%", "合同签订后 5 个工作日内"],
        ["到货款", "200000", "50%", "设备到货并验收合格"],
        ["质保金", "80000", "20%", "质保期满无质量问题"],
    ]
    tbl = Table(data, colWidths=[90, 90, 60, 180])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    doc.build([tbl])
    print("已生成:", path)


def main():
    _generate_table_pdf(OUT)

    pdf = pdfium.PdfDocument(OUT)
    page = pdf[0]
    bitmap = page.render(scale=2.0)
    img = np.asarray(bitmap.to_pil().convert("L"))
    page.close()
    bitmap.close()
    pdf.close()

    print("图片尺寸:", img.shape)

    table = table_pipeline.extract_table(img)
    print("\n===== 表格结构化结果 =====")
    if not table:
        print("（未识别出表格）")
        return
    for row in table:
        print(" | ".join(c if c else "·" for c in row))
    print("\n===== 制表符文本 =====")
    print(table_pipeline.table_to_text(table))


if __name__ == "__main__":
    main()
