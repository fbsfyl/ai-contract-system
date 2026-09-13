"""合同分类（考核 B2）。LLM + 规则双通道，输出置信度。

- 通道①（LLM 语义）：用分类提示词让 LLM 判断类型 + 置信度，处理语义模糊文本。
- 通道②（规则关键词）：关键词命中统计，作为 LLM 失败/低置信度时的快速兜底。
"""
import logging

from app import llm
from app import prompts
from app.schemas import CONTRACT_TYPES

logger = logging.getLogger(__name__)

# 关键词规则（双通道的规则通道）
_RULE_KEYWORDS = {
    "采购": ["采购", "买方", "供应商", "交货", "到货"],
    "销售": ["销售", "卖方", "购买方", "出售"],
    "服务": ["服务", "委托", "开发", "咨询", "劳务"],
    "租赁": ["租赁", "出租", "承租", "租金", "租期"],
}


def classify_by_rules(text: str) -> tuple[str | None, float]:
    """规则通道：关键词命中统计。返回 (类型, 置信度)；无命中返回 (None, 0.0)。"""
    scores = {t: sum(text.count(kw) for kw in kws) for t, kws in _RULE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None, 0.0
    total = sum(scores.values())
    confidence = min(0.9, 0.5 + (scores[best] / max(total, 1)) * 0.4)
    return best, round(confidence, 3)


def _llm_classify(text: str) -> tuple[str, float]:
    """LLM 通道：返回 (合同类型, 置信度)。类型限定在枚举内。"""
    messages = prompts.build_classify_messages(text)
    data = llm.chat_json(messages)
    ctype = str(data.get("contract_type", "其他")).strip()
    if ctype not in CONTRACT_TYPES:
        logger.warning("分类输出越界：%s，回退为「其他」", ctype)
        ctype = "其他"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return ctype, confidence


def classify(text: str) -> tuple[str, float]:
    """双通道分类：LLM 语义为主，规则关键词兜底。"""
    try:
        ctype, confidence = _llm_classify(text)
    except Exception:
        logger.exception("LLM 分类失败，回退规则通道")
        ctype, confidence = "其他", 0.0

    # LLM 低置信度时用规则通道兜底
    if confidence < 0.5:
        rule_type, rule_conf = classify_by_rules(text)
        if rule_type is not None:
            logger.info("LLM 低置信度（%.2f），规则兜底为「%s」", confidence, rule_type)
            return rule_type, rule_conf
    return ctype, confidence
