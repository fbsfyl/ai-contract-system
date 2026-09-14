"""线性流水线（考核 D3）：分类 → 提取，用 LangChain LCEL 串行编排。

选型说明（LangChain vs LangGraph 的边界）：
- 解析 → 分类 → 提取 → 入库 是**无状态、无分支、一遍跑完**的线性流程，
  用 LangChain 的 LCEL（RunnableLambda 串行管道）即可，无需 LangGraph 的状态机。
- 审查 Agent（app/review_agent.py）是**有状态、有回路、需要中断恢复**的场景，
  才用 LangGraph（图编排 + checkpoint）。

两者各司其职：线性用 LangChain，回路用 LangGraph。
"""
import logging
import time

from langchain_core.runnables import RunnableLambda

from app import classifier, extractor

logger = logging.getLogger(__name__)


def _classify(text: str) -> dict:
    """线性步骤①：合同分类（LLM + 规则双通道）。"""
    start = time.perf_counter()
    ctype, confidence = classifier.classify(text)
    logger.info("[流水线] 分类：%s（置信度 %.2f，%.1fms）", ctype, confidence, (time.perf_counter() - start) * 1000)
    return {"text": text, "contract_type": ctype, "confidence": confidence}


def _extract(state: dict) -> dict:
    """线性步骤②：结构化提取（RAG few-shot + 校验重试）。"""
    start = time.perf_counter()
    contract = extractor.extract(state["text"], use_rag=True)
    logger.info("[流水线] 提取：type=%s 金额=%s（%.1fms）", contract.contract_type, contract.amount, (time.perf_counter() - start) * 1000)
    return {
        "contract": contract,
        "contract_type": state["contract_type"],
        "confidence": state["confidence"],
    }


# LCEL 串行管道：分类 → 提取
pipeline = RunnableLambda(_classify) | RunnableLambda(_extract)


def run(text: str) -> dict:
    """执行线性流水线，返回 {contract, contract_type, confidence}。"""
    return pipeline.invoke(text)
