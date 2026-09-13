"""一键跑通全部验证脚本（考核 G 的「测试用例」）。

用法：
    python -B run_tests.py           # 核心准确率 / 验证脚本
    python -B run_tests.py --all     # 额外跑 RAGAS 评估（较重，需联网）

每个脚本独立运行，任一失败会以非零退出码结束。
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CORE = [
    ("seed_examples.py", "写入向量库范例（RAG 检索库）"),
    ("evaluate.py", "few-shot vs 零样本准确率对比（D2）"),
    ("eval_ocr.py", "文字版 vs 扫描版准确率对比（E）"),
    ("test_agents.py", "多智能体接力验证（D5）"),
    ("test_table.py", "复杂表格管线验证（E）"),
]

EXTRA = [
    ("ragas_eval.py", "RAGAS 4 指标评估（D4，较重）"),
]


def run(name: str, desc: str) -> bool:
    print("\n" + "=" * 70)
    print(f"> {desc}  [{name}]")
    print("=" * 70)
    proc = subprocess.run(
        [sys.executable, "-B", str(BASE_DIR / "scripts" / name)],
        cwd=BASE_DIR,
    )
    ok = proc.returncode == 0
    print(("OK  " if ok else "FAIL") + f"  {name}  (exit {proc.returncode})")
    return ok


def main() -> None:
    scripts = CORE + (EXTRA if "--all" in sys.argv else [])
    results = [run(name, desc) for name, desc in scripts]
    passed = sum(results)
    print("\n" + "=" * 70)
    print(f"通过 {passed}/{len(results)}")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
