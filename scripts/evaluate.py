"""提取准确率对比评估（考核 D2 的核心拉分点：无 few-shot vs 有 few-shot）。

用法：python scripts/evaluate.py
会在测试集上分别跑 零样本(zero-shot) 与 RAG few-shot，输出逐字段准确率对比。
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import extractor  # noqa: E402

TESTS_PATH = os.path.join(BASE_DIR, "data", "tests.json")

FIELDS = [
    "contract_name", "contract_no", "contract_type", "party_a", "party_b",
    "amount", "amount_capital", "sign_date", "term_start", "term_end",
    "payment_method", "breach_liability", "dispute_resolution",
]


def _norm(s: str) -> str:
    return "".join(str(s).split()).lower()


def _norm_date(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def field_equal(key: str, pred, gold) -> bool:
    """逐字段核对：金额按数值、日期按数字序列语义等价，其余归一化字符串相等。"""
    pred = pred or ""
    gold = gold or ""
    if key == "amount":
        try:
            return abs(float(pred) - float(gold)) < 0.5
        except (TypeError, ValueError):
            return False
    if key in ("sign_date", "term_start", "term_end"):
        a, b = _norm_date(pred), _norm_date(gold)
        return bool(a) and a == b
    return bool(pred) and _norm(pred) == _norm(gold)


def compare(pred: dict, gold: dict) -> tuple[int, int]:
    correct = sum(1 for k in FIELDS if field_equal(k, pred.get(k), gold.get(k)))
    return correct, len(FIELDS)


def main():
    with open(TESTS_PATH, encoding="utf-8") as f:
        tests = json.load(f)

    results = {}
    for mode, use_rag in [("zero_shot", False), ("few_shot", True)]:
        total_correct = 0
        total_fields = 0
        per_test = []
        for t in tests:
            pred = extractor.extract(t["text"], use_rag=use_rag)
            c, tot = compare(pred.model_dump(), t["fields"])
            total_correct += c
            total_fields += tot
            per_test.append((t["id"], c, tot))
        acc = total_correct / total_fields * 100 if total_fields else 0.0
        results[mode] = (acc, per_test)
        print(f"[{mode}] 整体准确率：{acc:.1f}%  ({total_correct}/{total_fields})")
        for tid, c, tot in per_test:
            print(f"    - {tid}: {c}/{tot}")

    zero_acc = results["zero_shot"][0]
    few_acc = results["few_shot"][0]
    delta = few_acc - zero_acc
    print("\n========== 对比结论 ==========")
    print(f"无 few-shot：{zero_acc:.1f}%")
    print(f"有 few-shot：{few_acc:.1f}%")
    print(f"提升：{delta:+.1f}%  {'↑ RAG few-shot 有效' if delta > 0 else '↓ 未见提升，检查范例质量/检索相似度'}")


if __name__ == "__main__":
    main()
