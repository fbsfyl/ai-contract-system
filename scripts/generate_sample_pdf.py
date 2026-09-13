"""生成「已填好」的文字版合同 PDF，用于测试上传链路。

数据源：data/tests.json 中的 test_purchase_1（服务器采购合同，金额/日期/双方全填好）。
用法：python scripts/generate_sample_pdf.py
"""
import json
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_PATH = os.path.join(BASE_DIR, "data", "tests.json")
OUT_PATH = os.path.join(BASE_DIR, "data", "filled_contract.pdf")

FONT = "STSong-Light"  # Adobe 内置中文宋体，无需外部字体文件


def main():
    with open(TESTS_PATH, encoding="utf-8") as f:
        tests = json.load(f)

    target = next(t for t in tests if t["id"] == "test_purchase_1")
    text = target["text"]

    pdfmetrics.registerFont(UnicodeCIDFont(FONT))

    c = canvas.Canvas(OUT_PATH, pagesize=A4)
    width, height = A4

    c.setFont(FONT, 12)
    y = height - 40 * mm
    line_height = 16

    for line in text.split("\n"):
        if y < 40 * mm:  # 换页
            c.showPage()
            c.setFont(FONT, 12)
            y = height - 40 * mm
        c.drawString(25 * mm, y, line)
        y -= line_height

    c.save()
    print("已生成:", OUT_PATH)


if __name__ == "__main__":
    main()
