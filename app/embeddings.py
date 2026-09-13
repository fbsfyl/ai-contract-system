"""Embedding 封装（考核 D1）。

支持两种 provider（通过 .env 的 EMBEDDING_PROVIDER 切换）：
- local：本地 sentence-transformers（BAAI/bge-small-zh-v1.5），无需额外 API key。
- api  ：OpenAI 兼容 embedding 接口（硅基流动 / 智谱 / 阿里百炼等）。

统一对外接口：embed(texts: list[str]) -> list[list[float]]
"""
import logging

from app import config

logger = logging.getLogger(__name__)

_provider = None


def _local_embed(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    global _provider
    if _provider is None:
        logger.info("加载本地 Embedding 模型：%s", config.LOCAL_EMBEDDING_MODEL)
        _provider = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    vectors = _provider.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def _api_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    global _provider
    if _provider is None:
        if not config.EMBEDDING_API_KEY:
            raise RuntimeError("EMBEDDING_PROVIDER=api 但未配置 EMBEDDING_API_KEY")
        _provider = OpenAI(api_key=config.EMBEDDING_API_KEY, base_url=config.EMBEDDING_BASE_URL)
    resp = _provider.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本转为向量。"""
    if not texts:
        return []
    if config.EMBEDDING_PROVIDER == "api":
        return _api_embed(texts)
    return _local_embed(texts)
