"""业财一体化（考核 B5 天花板）：合同 → 收付款计划 → 台账联动。

核心思想：合同是业财一体化的依据，收付款节点藏在「付款方式」字段里。
提取后按合同类型判定甲方应收/应付方向，生成结构化收付款计划并落库联动。
"""
import logging

from app import llm
from app import store

logger = logging.getLogger(__name__)

# 合同类型 → 甲方（我方）收付款方向
_DIRECTION_MAP = {
    "采购": "应付",  # 甲方是买方，向乙方付款
    "服务": "应付",  # 甲方采购服务，向乙方付款
    "销售": "应收",  # 甲方是卖方，向乙方收款
    "租赁": "应收",  # 甲方出租，向乙方收租金
    "其他": "应收",
}

FINANCE_SYSTEM_PROMPT = """你是业财一体化专家。请从合同「付款方式」中提取分期收付款节点。

规则：
- 识别付款分成几期（预付款/进度款/尾款/全款等）。
- 若写了一次性付款，输出 1 期，ratio=1.0，amount=合同总金额。
- 每期 amount = 合同总金额 × ratio（你自行计算）。
- 提取每期的触发条件（如「签订后」「验收合格后」「到货后」）。
- 若无信息，condition 填「未注明」。

只输出一个 JSON 对象，不要输出任何多余文字：
{"installments": [{"stage": "预付款", "ratio": 0.5, "amount": 240000, "condition": "合同签订后"}]}
"""


def direction_for(contract_type: str) -> str:
    return _DIRECTION_MAP.get(contract_type, "应收")


def extract_installments(payment_method: str, amount: float) -> list[dict]:
    """从付款方式字段提取分期节点（含金额/比例/触发条件）。"""
    user = f"合同总金额：{amount} 元\n付款方式：{payment_method}"
    data = llm.chat_json([
        {"role": "system", "content": FINANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])
    return _normalize(data.get("installments", []), amount)


def _normalize(items: list[dict], amount: float) -> list[dict]:
    """兜底：金额缺失用比例算，比例缺失用金额反推，避免脏数据进台账。"""
    out = []
    for it in items:
        ratio = float(it.get("ratio", 0) or 0)
        amt = float(it.get("amount", 0) or 0)
        if amt == 0 and ratio > 0:
            amt = round(amount * ratio, 2)
        out.append({
            "stage": it.get("stage", "付款"),
            "ratio": ratio,
            "amount": amt,
            "condition": it.get("condition", "未注明"),
        })

    # 若全都没给比例但给了金额，按金额反推比例
    if out and all(x["ratio"] == 0 for x in out) and amount > 0:
        for x in out:
            x["ratio"] = round(x["amount"] / amount, 4)

    # 兜底：什么都没提取到时，生成一期全款
    if not out:
        out = [{"stage": "全款", "ratio": 1.0, "amount": amount, "condition": "未注明"}]

    return out


def generate_plan(contract_id: int, contract_type: str, payment_method: str, amount: float) -> list[dict]:
    """生成收付款计划并写入台账联动表（重复生成先清空旧计划）。"""
    direction = direction_for(contract_type)
    installments = extract_installments(payment_method, amount)

    store.clear_plans(contract_id)
    for it in installments:
        store.insert_plan(
            contract_id=contract_id,
            direction=direction,
            stage=it["stage"],
            ratio=it["ratio"],
            amount=it["amount"],
            condition=it["condition"],
        )
    logger.info("合同 %d 生成 %d 期%s计划", contract_id, len(installments), direction)
    return store.list_plans(contract_id)
