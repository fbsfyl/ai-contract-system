"""提取流水线（考核 B1 + D2 灵魂）。

链路：文本 → (RAG 检索相似范例 → few-shot 注入) → LLM 提取 → Schema 校验 → 重试兜底。

这里实现了「把 RAG 当监督学习用」：向量库中预置类型正确的标准范例，
新合同进来先相似检索，把范例当 few-shot 参照喂给 LLM，提升提取准确率。
"""
import json
import logging

from app import llm
from app import prompts
from app import vector_store
from app.schemas import CONTRACT_TYPES, ContractFields, ExtractedContract

logger = logging.getLogger(__name__)


def _format_examples(examples: list[dict]) -> str:
    """把检索到的范例格式化为 few-shot 文本块。"""
    blocks = []
    for i, ex in enumerate(examples, 1):
        fields = json.dumps(ex["fields"], ensure_ascii=False)
        blocks.append(
            f"【范例{i}】类型：{ex['contract_type']}\n"
            f"文本片段：{ex['text'][:400]}\n"
            f"正确字段：{fields}"
        )
    return "\n\n".join(blocks)


def _normalize(data: dict) -> ContractFields:
    """宽松预处理 + 严格枚举/数值约束（越界触发重试，对应考核 A 的兜底逻辑）。"""
    # 金额可能是字符串（含千分位逗号），统一转 float
    raw_amount = str(data.get("amount", 0)).replace(",", "").replace("，", "").strip()
    try:
        data["amount"] = float(raw_amount)
    except (ValueError, TypeError):
        raise ValueError("amount 无法解析为数字")

    ctype = str(data.get("contract_type", "")).strip()
    if ctype not in CONTRACT_TYPES:
        raise ValueError(f"contract_type 越界：{ctype}，只能是 {CONTRACT_TYPES} 之一")

    return ContractFields(**data)


def extract(text: str, use_rag: bool = True, feedback: str = "") -> ExtractedContract:
    """从合同文本提取结构化字段。use_rag=False 时走零样本（用于对比实验）。

    feedback 为多智能体校验 Agent 回填的错误清单，注入后可让 LLM 自我修正重来。
    """
    reference_examples: list[str] = []
    few_shot = ""

    if use_rag:
        examples = vector_store.search_similar(text)
        reference_examples = [ex["id"] for ex in examples]
        few_shot = _format_examples(examples)
        logger.info("RAG 检索到 %d 条参照范例：%s", len(examples), reference_examples)

    messages = prompts.build_extract_messages(text, few_shot)
    if feedback:
        messages.append(
            {
                "role": "user",
                "content": "上一次抽取结果校验未通过，请按以下问题修正后重新抽取：\n" + feedback,
            }
        )

    # Schema 校验失败时，把错误回填给 LLM 重试（重试兜底）
    for attempt in range(3):
        data = llm.chat_json(messages)
        try:
            fields = _normalize(data)
            return ExtractedContract(
                **fields.model_dump(),
                reference_examples=reference_examples,
            )
        except Exception as e:
            logger.warning("字段校验失败（第 %d 次）：%s", attempt + 1, e)
            messages.append(
                {"role": "user", "content": f"你输出的字段有误：{e}。请修正后严格只输出一个合法 JSON 对象。"}
            )

    raise ValueError("多次提取均未通过字段校验")
