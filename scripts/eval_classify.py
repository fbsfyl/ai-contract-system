"""分类准确率评估（考核 B2：10 份样本盲测 ≥90%）。

用 5 条范例 + 10 条测试金标准共 15 份样本，对比：
- 规则通道（本地关键词，快）
- LLM 通道（语义分类，主通道）

用法：python -B scripts/eval_classify.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import classifier  # noqa: E402

EXAMPLES_PATH = os.path.join(BASE_DIR, "data", "examples", "contracts.json")
TESTS_PATH = os.path.join(BASE_DIR, "data", "tests.json")


def load_samples() -> list[tuple[str, str, str]]:
    samples = []
    for path in (EXAMPLES_PATH, TESTS_PATH):
        with open(path, encoding="utf-8") as f:
            for it in json.load(f):
                samples.append((it["id"], it["contract_type"], it["text"]))
    return samples


def main() -> None:
    samples = load_samples()

    rule_correct = 0
    llm_correct = 0
    print("=" * 70)
    print(f"分类准确率评估（{len(samples)} 份样本）")
    print("=" * 70)
    print(f"{'样本':<24}{'金标准':<8}{'规则':<8}{'LLM':<8}LLM置信度")
    for sid, gold, text in samples:
        rule_type, _ = classifier.classify_by_rules(text)
        rule_type = rule_type or "其他"
        llm_type, llm_conf = classifier.classify(text)

        rule_ok = rule_type == gold
        llm_ok = llm_type == gold
        rule_correct += rule_ok
        llm_correct += llm_ok
        mark = ("  *规则错" if not rule_ok else "") + ("  *LLM错" if not llm_ok else "")
        print(f"{sid:<24}{gold:<8}{rule_type:<8}{llm_type:<8}{llm_conf:.2f}{mark}")

    n = len(samples)
    print("\n========== 结果 ==========")
    print(f"规则通道：{rule_correct}/{n} = {rule_correct / n * 100:.1f}%")
    print(f"LLM 通道：{llm_correct}/{n} = {llm_correct / n * 100:.1f}%")
    print("\nB2 要求：10 份样本分类准确率 ≥90%")
    print("结论：" + ("达标" if llm_correct / n >= 0.9 else "未达标，需检查分类提示词/规则关键词"))


if __name__ == "__main__":
    main()
