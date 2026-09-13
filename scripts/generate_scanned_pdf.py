"""生成扫描版合同 PDF：把文字版合同渲染成图片再合成，模拟「扫描件」场景。

用于测试 OCR 管线（考核 E 分流：文字版直接抽文本，扫描件走 RapidOCR）。
用法：python scripts/generate_scanned_pdf.py
"""
import os

import pypdfium2 as pdfium

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "data", "filled_contract.pdf")
OUT = os.path.join(BASE_DIR, "data", "scanned_contract.pdf")


def main():
    pdf = pdfium.PdfDocument(SRC)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=2.0)  # 放大提升清晰度
        images.append(bitmap.to_pil().convert("RGB"))
        page.close()
        bitmap.close()
    pdf.close()

    # PIL 多页 PDF：图片型 PDF（无文字层，pdfplumber 抽不到文本 → 判定为扫描件）
    images[0].save(OUT, "PDF", save_all=True, append_images=images[1:], resolution=100.0)
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
