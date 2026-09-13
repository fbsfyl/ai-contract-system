"""PDF 加载与场景分流（考核 E）。

核心：不是无脑上 OCR。
- 文字版 PDF（能抽出文本）→ pdfplumber 直接抽取，根本不需要 OCR。
- 扫描件（抽不出文本）→ 返回空并标记需要 OCR（本最小集只实现文字版，
  扫描件给出明确提示，预留 OCR 管线接入位）。
"""
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> tuple[str, str]:
    """返回 (文本, 场景)。场景为 'text' 或 'scanned'。

    分流逻辑：尝试用 pdfplumber 抽取，若有效字符过少则判定为扫描件。
    """
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    text = "\n".join(pages_text).strip()
    # 判定：有效中文字符少于阈值视为扫描件（图片型 PDF）
    meaningful = len([c for c in text if c.isalnum() or "\u4e00" <= c <= "\u9fff"])
    if meaningful < 20:
        logger.warning("PDF 有效文本过少（%d 字符），判定为扫描件，需走 OCR 管线", meaningful)
        return "", "scanned"
    return text, "text"
