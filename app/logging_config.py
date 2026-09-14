"""统一日志配置（考核 G 工程化）：控制台 + 滚动文件双通道，记录应用内部全流程。

用法：在 app 入口（app/main.py）最早处调用 setup_logging()，之后各模块用
logging.getLogger(__name__) 拿到的 logger 会同时输出到控制台与 logs/app.log。

设计要点：
- 滚动文件：logs/app.log，单文件 5MB，最多保留 5 个历史（app.log.1 ~ app.log.5）。
- 统一格式：时间 + 级别 + 模块名 + 消息，方便按模块回溯流程。
- 降噪：chromadb / httpx / uvicorn 等第三方库日志压到 WARNING，只保留本项目 app.* 的 INFO。
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app import config

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 第三方库日志较吵，压到 WARNING，只突出本项目 app.* 的 INFO
_NOISY_LOGGERS = (
    "chromadb",
    "chromadb.api",
    "httpx",
    "httpcore",
    "urllib3",
    "uvicorn.access",
    "uvicorn.error",
    "sentence_transformers",
    "rapidocr",
    "pypdfium2",
    "langchain_core",
    "langchain",
)

_configured = False


def setup_logging() -> Path:
    """配置根 logger：控制台 + logs/app.log（滚动）。返回日志文件路径。"""
    global _configured

    log_dir = config.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    if _configured:
        return log_file

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return log_file
