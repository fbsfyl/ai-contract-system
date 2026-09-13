"""LLM 客户端封装（DeepSeek，OpenAI 兼容协议）。

封装原因（考核 C 的「技术解耦」）：上层只依赖本模块的 chat_json，
未来替换为 OpenAI / Qwen / 本地模型时，只改这里的 base_url 与 model 即可。
"""
import json
import logging

from openai import OpenAI

from app import config

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 中设置后重试")
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


def chat_json(
    messages: list[dict],
    temperature: float = 0.0,
    max_retries: int = 2,
) -> dict:
    """调用 LLM 并强制返回 JSON 对象。

    max_retries 对应「解析失败自动重试兜底逻辑」：LLM 偶发输出非 JSON 时，
    加一句更严格的指令重试，避免单次失败直接崩掉整条链路。
    """
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = get_client().chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            last_err = e
            logger.warning("LLM 输出解析失败，第 %d 次重试：%s", attempt + 1, e)
            messages.append(
                {"role": "user", "content": "你上次的输出不是合法 JSON，请严格只输出一个 JSON 对象。"}
            )
        except Exception as e:  # 网络/鉴权等不重试，直接抛出
            raise e

    raise ValueError(f"LLM 多次输出仍无法解析为 JSON：{last_err}")
