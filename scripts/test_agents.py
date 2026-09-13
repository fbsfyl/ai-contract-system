"""验证多智能体接力流水线（考核 D5）：分类 → 提取 → 校验 → 审查。

用法：python scripts/test_agents.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import agents  # noqa: E402

TESTS_PATH = os.path.join(BASE_DIR, "data", "tests.json")


def main():
    with open(TESTS_PATH, encoding="utf-8") as f:
        items = json.load(f)

    for it in items:
        print("=" * 60)
        print(f"测试：{it['id']}（金标准类型：{it['contract_type']}）")
        result = agents.run_agents(it["text"])

        print(f"分类 Agent → {result['contract_type']}（置信度 {result['confidence']}）")
        print(f"提取 Agent → {result['contract']['contract_name']}（参照范例 {result['reference_examples']}）")
        print(f"提取次数：{result['extract_attempts']}，残错：{result['residual_errors']}")
        print(f"审查 Agent → {result['verdict']}，风险 {len(result['risks'])} 条")

        print("\n各 Agent 轨迹：")
        for step in result["trace"]:
            print("  -", json.dumps(step, ensure_ascii=False))


if __name__ == "__main__":
    main()
