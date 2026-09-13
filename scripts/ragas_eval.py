"""RAGAS 评估（考核 D2 进阶）：忠实度 / 上下文精度 / 上下文召回。

用途：在本系统「把 RAG 当监督学习用」的检索增强链路之上，用 RAGAS 量化检索与生成质量。

指标口径（针对本系统做适配）：
- 忠实度 faithfulness：抽取结果（answer）与检索到的参照范例（contexts）的一致程度。
  说明：字段值本身来自合同正文，因此该指标更多反映「抽取是否遵循范例的字段规范口径」，
  逐字段准确率仍以 scripts/evaluate.py 为准。
- 上下文精度 context_precision：检索到的范例与待抽取合同（question）的相关程度。
- 上下文召回 context_recall：金标准字段（ground_truth）所需信息被检索到的范例覆盖的程度。

用法：python scripts/ragas_eval.py
"""
import json
import os
import sys
import types
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

warnings.filterwarnings("ignore")

# 关闭 ragas 遥测，避免往用户目录写 uuid/上报
os.environ["RAGAS_DO_NOT_TRACK"] = "True"


def _patch_langchain_vertexai() -> None:
    """修复 ragas 0.4.x 与 langchain-community 0.4.x 的兼容问题。

    ragas 的 ragas/llms/base.py 仍从 langchain_community 导入已迁移走的
    VertexAI 集成（0.4 起被拆分到 langchain-google-vertexai）。本系统只用
    OpenAI 兼容的 DeepSeek，不依赖 VertexAI，因此注册占位模块即可。
    """
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
    sys.modules["langchain_community.chat_models.vertexai"] = mod

    import langchain_community.chat_models as _chat_models
    import langchain_community.llms as _llms

    _chat_models.vertexai = mod
    if not hasattr(_llms, "VertexAI"):
        _llms.VertexAI = type("VertexAI", (), {})  # type: ignore[attr-defined]


_patch_langchain_vertexai()


def _disable_ragas_analytics() -> None:
    """阻止 ragas 事件默认工厂去创建用户目录 uuid.json。"""
    import ragas._analytics as _analytics

    # 覆盖所有事件类的 user_id 默认工厂，避免落盘
    for obj in vars(_analytics).values():
        if isinstance(obj, type) and issubclass(obj, _analytics.BaseEvent):
            if "user_id" in getattr(obj, "model_fields", {}):
                obj.model_fields["user_id"].default_factory = (
                    lambda: "anonymous-user"
                )


_disable_ragas_analytics()

from datasets import Dataset  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from app import config, embeddings, extractor, vector_store  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

TESTS_PATH = os.path.join(BASE_DIR, "data", "tests.json")
EXAMPLES_PATH = os.path.join(BASE_DIR, "data", "examples", "contracts.json")


class LocalEmbeddings(Embeddings):
    """把本系统的 sentence-transformers 适配成 LangChain Embeddings 接口。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embeddings.embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embeddings.embed([text])[0]


def _ensure_seeded() -> None:
    """向量库为空时先写入标准范例，保证检索有上下文。"""
    if vector_store.count() > 0:
        return
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        items = json.load(f)
    vector_store.add_examples(
        [
            {
                "id": it["id"],
                "text": it["text"],
                "metadata": {
                    "contract_type": it["contract_type"],
                    "fields": json.dumps(it["fields"], ensure_ascii=False),
                },
            }
            for it in items
        ]
    )
    print(f"向量库为空，已自动写入 {len(items)} 条标准范例")


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.0,
    )


def _build_samples() -> list[dict]:
    with open(TESTS_PATH, encoding="utf-8") as f:
        tests = json.load(f)

    rows = []
    for t in tests:
        text = t["text"]
        # 运行真实链路：检索参照范例 + 抽取字段
        reference = vector_store.search_similar(text)
        answer = extractor.extract(text, use_rag=True).model_dump()
        contexts = [ex["text"] for ex in reference]
        rows.append(
            {
                "question": text,
                "answer": json.dumps(answer, ensure_ascii=False),
                "contexts": contexts,
                "ground_truth": json.dumps(t["fields"], ensure_ascii=False),
            }
        )
    return rows


METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# 指标口径说明（写进评估报告）
METRIC_DESC = {
    "faithfulness": "抽取结果(answer)与检索到的参照范例(contexts)的一致程度；本系统字段值来自正文，故更多反映「是否遵循范例字段口径」",
    "answer_relevancy": "抽取结果(answer)是否紧扣待抽取合同(question)，衡量生成内容与原文的相关性",
    "context_precision": "检索到的范例与待抽取合同(question)的相关程度（检索层精度）",
    "context_recall": "金标准字段(ground_truth)所需信息被检索范例覆盖的程度（检索层召回）",
}

LOW_SCORE_THRESHOLD = 0.5


def _build_report(means: dict[str, float], scores: list[dict]) -> list[str]:
    """生成评估报告正文（含低分项诊断与改进措施）。"""
    lines = [
        "RAGAS 评估报告",
        "=" * 40,
        "评估对象：合同系统 RAG few-shot 链路（检索参照范例 → LLM 结构化提取）",
        "",
        "一、聚合分",
    ]
    for name in METRIC_NAMES:
        lines.append(f"  {name:<18} {means[name]:.4f}  | {METRIC_DESC[name]}")
    lines.append("")
    lines.append("二、逐条明细")
    for i, row_scores in enumerate(scores):
        line = "  ".join(f"{k}={v:.3f}" for k, v in row_scores.items())
        lines.append(f"  sample_{i}: {line}")

    low = {k: v for k, v in means.items() if v < LOW_SCORE_THRESHOLD}
    lines.append("")
    lines.append(f"三、低分项诊断（阈值 < {LOW_SCORE_THRESHOLD}）")
    if not low:
        lines.append("  无低分项。")
    else:
        for k, v in low.items():
            lines.append(f"  - {k}: {v:.4f}")
        lines.append("")
        lines.append("四、改进措施（针对低分项的一轮改进）")
        if "context_precision" in low or "context_recall" in low:
            lines.append(
                "  · 检索层：整份范例作为检索单元粒度过粗，导致 context_precision/recall 偏低；"
                "已改为「按条款切块检索 + 相似度分数」，见 app/vector_store.py 的 split_contract_by_clause / search_chunks。"
            )
        if "faithfulness" in low:
            lines.append(
                "  · 生成层：answer 为完整字段 JSON 而 context 为范例全文，口径不一致；"
                "以 scripts/evaluate.py 的逐字段准确率为准，RAGAS 分仅作检索/相关性参考。"
            )
        if "answer_relevancy" in low:
            lines.append(
                "  · 相关性：抽取目标字段与原文的相关性低时，优先扩充同类标准范例、强化 few-shot 注入。"
            )
    return lines


def main() -> None:
    _ensure_seeded()
    rows = _build_samples()

    result = evaluate(
        Dataset.from_dict(
            {
                "question": [r["question"] for r in rows],
                "answer": [r["answer"] for r in rows],
                "contexts": [r["contexts"] for r in rows],
                "ground_truth": [r["ground_truth"] for r in rows],
            }
        ),
        metrics=METRICS,
        llm=_build_llm(),
        embeddings=LocalEmbeddings(),
        show_progress=True,
    )

    import numpy as np

    means: dict[str, float] = {}
    for name in METRIC_NAMES:
        vals = result[name]
        means[name] = float(np.nanmean(vals)) if vals else float("nan")

    print("\n========== RAGAS 评估结果 ==========")
    for name in METRIC_NAMES:
        print(f"{name}: {means[name]:.4f}")

    print("\n逐条明细：")
    for i, row_scores in enumerate(result.scores):
        line = "  ".join(f"{k}={v:.3f}" for k, v in row_scores.items())
        print(f"  sample_{i}: {line}")

    report = _build_report(means, result.scores)
    report_path = os.path.join(BASE_DIR, "data", "ragas_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n评估报告已写入：{report_path}")


if __name__ == "__main__":
    main()
