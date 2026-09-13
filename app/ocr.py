"""扫描件 OCR 管线（考核 E 场景分流）。

文字版 PDF 直接抽文本（pdf_loader）；扫描件/图片型 PDF 走本模块：
PDF 页 → 渲染成高分辨率图片 → RapidOCR 识别 → 拼接文本。

选型说明：RapidOCR（PaddleOCR 的 ONNX 运行时版）无需 paddlepaddle 重型依赖，
内置中文识别模型，pip 安装即用；复杂表格/骑缝章等特殊场景可后续接 PP-Structure。
"""
import logging

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
        logger.info("RapidOCR 引擎初始化完成")
    return _engine


def _ocr_image(img) -> str:
    """识别单张图片（PIL Image 或 numpy array），返回文本。"""
    engine = _get_engine()
    result = engine(img)
    if result is None:
        return ""
    # rapidocr 3.x 返回 RapidOCROutput（带 .txts）；兼容旧版 [box, text, score] 列表
    if hasattr(result, "txts"):
        if not result.txts:  # 空单元格/无文字区域
            return ""
        return "\n".join(str(t) for t in result.txts)
    if not result:
        return ""
    return "\n".join(str(item[1]) for item in result)


def ocr_pdf(pdf_path: str, scale: float = 2.0) -> str:
    """对扫描版 PDF 逐页 OCR，返回拼接文本。"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    page_texts = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)  # 放大提升识别率
            img = bitmap.to_pil()              # PIL Image
            text = _ocr_image(img)
            if text:
                page_texts.append(text)
            page.close()
            bitmap.close()
    finally:
        pdf.close()
    return "\n\n".join(page_texts)
