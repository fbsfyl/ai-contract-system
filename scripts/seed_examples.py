"""把标准范例写入向量库（考核 D2 的「预置类型正确的标注范例集」）。

用法：python scripts/seed_examples.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import vector_store  # noqa: E402

EXAMPLES_PATH = os.path.join(BASE_DIR, "data", "examples", "contracts.json")


def main():
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        items = json.load(f)

    prepared = [
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

    vector_store.add_examples(prepared)

    # 同时按条款切块入库（考核 D1 的切块 + 检索可视化）
    chunk_total = 0
    for it in items:
        chunk_total += vector_store.index_chunks(it["id"], it["contract_type"], it["text"])

    print(
        f"已写入 {len(prepared)} 条范例，向量库总数：{vector_store.count()}，"
        f"条款块：{chunk_total}（共 {vector_store.chunk_count()}）"
    )


if __name__ == "__main__":
    main()
