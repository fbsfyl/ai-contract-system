"""多智能体接力流水线（考核 D5）。

四个 Agent 接力完成任务，出错能自我修正重来：
    分类 Agent → 提取 Agent → 校验 Agent → 审查 Agent

用 LangGraph 编排，体现「规划 → 工具调用 → 反思迭代」的多 Agent 机制：
- 规划：图中节点顺序即任务分解（先分类定调，再提取，再校验，最后审查）。
- 工具调用：各 Agent 内部调用 LLM / 规则关键词 / 向量检索等工具。
- 反思迭代：校验 Agent 捕获错误 → 把错误回填给提取 Agent → 重新提取（自我修正回路）。

与 app/pipeline.py（LangChain LCEL 线性管道）的区别：
- pipeline.py 无状态、无分支、一遍跑完，适合「解析→提取→入库」线性任务（考核 D3）。
- 本模块有状态、有条件分支、有回路，适合需要「校验失败回头重做」的多 Agent 任务（考核 D5）。
"""
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app import classifier, extractor, review_agent

logger = logging.getLogger(__name__)

MAX_EXTRACT_RETRIES = 2  # 校验失败最多重提取次数，超过则带残错进入审查

# 校验 Agent 关注的关键字段（合同「灵魂」字段，缺失即判不合格）
_REQUIRED_FIELDS = {
    "contract_name": "合同名称",
    "party_a": "甲方",
    "party_b": "乙方",
    "sign_date": "签订日期",
    "payment_method": "付款方式",
    "breach_liability": "违约责任",
    "dispute_resolution": "争议解决",
}


class AgentState(TypedDict):
    text: str
    contract_type: str
    confidence: float
    contract: dict
    reference_examples: list[str]
    errors: list[str]
    extract_attempts: int
    verdict: str
    risks: list[str]
    missing: list[str]
    trace: list[dict]


def _classify_node(state: AgentState) -> dict:
    """分类 Agent：判定合同类型，为后续提取/校验定调。"""
    ctype, confidence = classifier.classify(state["text"])
    trace = list(state.get("trace", [])) + [
        {"agent": "分类 Agent", "contract_type": ctype, "confidence": confidence}
    ]
    logger.info("分类 Agent → %s（置信度 %.2f）", ctype, confidence)
    return {"contract_type": ctype, "confidence": confidence, "trace": trace}


def _extract_node(state: AgentState) -> dict:
    """提取 Agent：RAG few-shot 提取结构化字段；带校验反馈时自我修正重来。"""
    feedback = "\n".join(state.get("errors", []))
    contract = extractor.extract(state["text"], use_rag=True, feedback=feedback)

    trace = list(state.get("trace", [])) + [
        {
            "agent": "提取 Agent",
            "contract_name": contract.contract_name,
            "contract_type": contract.contract_type,
            "reference_examples": contract.reference_examples,
            "attempt": state.get("extract_attempts", 0) + 1,
        }
    ]
    logger.info(
        "提取 Agent（第 %d 次）→ %s，参照范例 %s",
        state.get("extract_attempts", 0) + 1,
        contract.contract_name,
        contract.reference_examples,
    )
    return {
        "contract": contract.model_dump(),
        "reference_examples": contract.reference_examples,
        "extract_attempts": state.get("extract_attempts", 0) + 1,
        "errors": [],  # 清空旧错误，交由校验 Agent 重新评估
        "trace": trace,
    }


def _validate_contract(contract: dict, contract_type: str) -> list[str]:
    """校验 Agent 的业务规则：关键字段完备性 + 类型一致性。"""
    errors: list[str] = []

    for field, label in _REQUIRED_FIELDS.items():
        val = str(contract.get(field, "") or "").strip()
        if not val or val == "未注明":
            errors.append(f"{label}（{field}）缺失或为「未注明」")

    amount = contract.get("amount", 0) or 0
    if float(amount) <= 0:
        errors.append("合同金额（amount）缺失或为 0")

    extracted_type = contract.get("contract_type", "")
    if contract_type and extracted_type and extracted_type != contract_type:
        errors.append(
            f"分类 Agent 判定为「{contract_type}」，提取 Agent 输出「{extracted_type}」，需对齐"
        )

    return errors


def _validate_node(state: AgentState) -> dict:
    """校验 Agent：对提取结果做业务规则校验，产出错误清单。"""
    errors = _validate_contract(state.get("contract", {}), state.get("contract_type", ""))
    trace = list(state.get("trace", [])) + [
        {"agent": "校验 Agent", "errors": errors, "passed": not errors}
    ]
    logger.info("校验 Agent → %s", "通过" if not errors else f"{len(errors)} 处错误")
    return {"errors": errors, "trace": trace}


def _review_node(state: AgentState) -> dict:
    """审查 Agent：法务审查（复用 LangGraph 审查 Agent），给出最终结论。"""
    result = review_agent.review(state["text"])
    trace = list(state.get("trace", [])) + [
        {"agent": "审查 Agent", "verdict": result["verdict"], "risks": result["risks"]}
    ]
    logger.info("审查 Agent → %s", result["verdict"])
    return {
        "verdict": result["verdict"],
        "risks": result["risks"],
        "missing": result["missing"],
        "trace": trace,
    }


def _route_after_validate(state: AgentState) -> str:
    """反思迭代：校验失败且仍有重试预算时回提取 Agent 自我修正，否则进审查。"""
    if state.get("errors") and state.get("extract_attempts", 0) < MAX_EXTRACT_RETRIES:
        return "extract"
    return "review"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", _classify_node)
    g.add_node("extract", _extract_node)
    g.add_node("validate", _validate_node)
    g.add_node("review", _review_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"extract": "extract", "review": "review"},
    )
    g.add_edge("review", END)
    return g.compile()


_compiled = None


def run_agents(text: str) -> dict:
    """执行多智能体接力，返回最终状态（含各 Agent 轨迹、校验残错、审查结论）。"""
    global _compiled
    if _compiled is None:
        _compiled = _build_graph()

    result = _compiled.invoke({
        "text": text,
        "contract_type": "",
        "confidence": 0.0,
        "contract": {},
        "reference_examples": [],
        "errors": [],
        "extract_attempts": 0,
        "verdict": "",
        "risks": [],
        "missing": [],
        "trace": [],
    })

    return {
        "contract_type": result.get("contract_type"),
        "confidence": result.get("confidence"),
        "contract": result.get("contract", {}),
        "reference_examples": result.get("reference_examples", []),
        "residual_errors": result.get("errors", []),
        "extract_attempts": result.get("extract_attempts", 0),
        "verdict": result.get("verdict"),
        "risks": result.get("risks", []),
        "missing": result.get("missing", []),
        "trace": result.get("trace", []),
    }
