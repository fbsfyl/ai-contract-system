"""LangGraph 审查 Agent（考核 D3/D5 部分）。

用 LangGraph 构建有状态、有回路的审查流程：
审查 → 发现风险/缺料 → 补材料 → 重新审查 → 通过 / 拒绝 / 人工介入。

相比 LangChain 线性流水线，这里体现「状态机 + 条件分支 + 循环回路 + 中断恢复」。
"""
import logging

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app import llm

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2  # 最多补材料重审次数，超过转人工介入

REVIEW_SYSTEM_PROMPT = """你是资深法务，审查合同并给出结论。

请识别合同风险，并给出结论 verdict：
- pass：要素齐全、无重大风险，可通过
- supplement：缺少关键要素（如金额、违约条款、争议解决、履行期限等），需要补充后重审
- reject：存在重大合规/法律风险，应拒绝

只输出一个 JSON 对象，不要输出任何多余文字：
{"verdict": "pass", "risks": ["风险点1"], "missing": ["缺少的要素1"]}
risks 为发现的风险列表（无则空数组）；missing 为需要补充的要素列表（仅 verdict=supplement 时填写）。
"""


class ReviewState(TypedDict):
    text: str
    verdict: str
    risks: list[str]
    missing: list[str]
    iteration: int
    supplements: list[str]  # 已补充材料记录
    history: list[dict]     # 每轮审查结果，追踪回路


def _review_node(state: ReviewState) -> dict:
    """审查节点：调用 LLM 输出 verdict/risks/missing。"""
    extra = ""
    if state.get("supplements"):
        extra = (
            "\n\n【此前审查已识别的缺失项，重审时请判断是否仍然缺失】\n"
            + "\n".join(f"- {s}" for s in state["supplements"])
            + "\n若这些缺失项在本合同文本中仍无法确认，请继续判 supplement；"
            "若确已齐备、无实质缺失，可判 pass。"
        )

    messages = [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": "请审查以下合同：\n" + state["text"] + extra},
    ]
    data = llm.chat_json(messages)

    verdict = data.get("verdict", "supplement")
    risks = data.get("risks", [])
    missing = data.get("missing", [])

    history = list(state.get("history", []))
    history.append({
        "iteration": state.get("iteration", 0),
        "verdict": verdict,
        "risks": risks,
        "missing": missing,
    })

    return {"verdict": verdict, "risks": risks, "missing": missing, "history": history}


def _supplement_node(state: ReviewState) -> dict:
    """补材料节点：记录需补充的要素，iteration+1，形成回路准备。"""
    supplements = list(state.get("supplements", []))
    supplements.extend(state.get("missing", []))
    return {
        "supplements": supplements,
        "iteration": state.get("iteration", 0) + 1,
    }


def _human_node(state: ReviewState) -> dict:
    """人工介入节点：多次补料仍无法通过，标记需人工处理。"""
    return {"verdict": "human"}


def _route_after_review(state: ReviewState) -> str:
    verdict = state.get("verdict", "supplement")
    if verdict == "pass":
        return "pass"
    if verdict == "reject":
        return "reject"
    # supplement：若还有重审预算则补材料（回路），否则转人工
    if state.get("iteration", 0) < MAX_ITERATIONS:
        return "supplement"
    return "human"


def _build_graph():
    g = StateGraph(ReviewState)
    g.add_node("review", _review_node)
    g.add_node("supplement", _supplement_node)
    g.add_node("human", _human_node)

    g.add_edge(START, "review")
    g.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "pass": END,
            "reject": END,
            "supplement": "supplement",
            "human": "human",
        },
    )
    g.add_edge("supplement", "review")  # 回路：补料后重新审查
    g.add_edge("human", END)
    return g.compile()


_compiled = None


def review(text: str) -> dict:
    """执行审查，返回最终状态（含回路轨迹）。"""
    global _compiled
    if _compiled is None:
        _compiled = _build_graph()

    result = _compiled.invoke({
        "text": text,
        "verdict": "",
        "risks": [],
        "missing": [],
        "iteration": 0,
        "supplements": [],
        "history": [],
    })

    logger.info("审查完成：verdict=%s，回路次数=%d", result.get("verdict"), result.get("iteration", 0))
    return {
        "verdict": result.get("verdict"),
        "risks": result.get("risks", []),
        "missing": result.get("missing", []),
        "supplements": result.get("supplements", []),
        "iterations": result.get("iteration", 0),
        "history": result.get("history", []),
    }
