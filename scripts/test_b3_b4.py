"""B3 合同信息管理 + B4 商用加固 的端到端测试（HTTP 层）。

覆盖：登录鉴权、模板/条款库、自动编号、要素拼装起草、状态机流转、台账看板、CSV 导出。
用法：python -B scripts/test_b3_b4.py（使用临时库，不污染真实 contracts.db）
"""
import os
import sys
import tempfile
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import config  # noqa: E402

# 隔离到临时库，避免污染真实 contracts.db
config.DB_PATH = os.path.join(tempfile.mkdtemp(prefix="contract_b34_"), "test.db")

from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def main() -> None:
    year = datetime.now().year
    with TestClient(app) as client:
        # ---- B4 登录鉴权 ----
        r = client.post("/api/login", json={"username": "admin", "password": "wrong"})
        check("登录-错误密码返回401", r.status_code == 401, f"status={r.status_code}")

        r = client.post("/api/login", json={"username": config.ADMIN_USER, "password": config.ADMIN_PASSWORD})
        token = r.json().get("token", "") if r.status_code == 200 else ""
        check("登录-正确账号返回token", r.status_code == 200 and bool(token), f"status={r.status_code}")
        auth = {"Authorization": f"Bearer {token}"}

        r = client.post("/api/draft", json={})
        check("起草-未登录返回401", r.status_code == 401, f"status={r.status_code}")

        # ---- B3 模板/条款库 ----
        r = client.get("/api/templates")
        types = [t["type"] for t in r.json()]
        check("模板-含5类模板", set(types) == {"其他", "采购", "服务", "租赁", "销售"}, str(types))
        check("模板-采购含条款库", len(r.json()[0]["clauses"]["breach_liability"]) == 2, "")

        # ---- B3 自动编号 ----
        r = client.get("/api/contracts/next_no", params={"contract_type": "采购"})
        no1 = r.json()["contract_no"]
        check("编号-首号格式", no1 == f"HT-CG-{year}-001", no1)

        # ---- B3 要素拼装起草 ----
        payload = {
            "contract_type": "采购",
            "contract_name": "测试服务器采购合同",
            "party_a": "南京某某云计算有限公司",
            "party_b": "苏州某某信息技术有限公司",
            "amount": 480000,
            "sign_date": "2025-02-20",
            "term_start": "2025-02-20",
            "term_end": "2025-03-31",
            "payment_method": "合同签订后预付50%，验收合格后支付50%",
            "breach_liability": "乙方逾期交货的，每逾期一日按合同金额的0.5%支付违约金",
            "dispute_resolution": "双方协商不成的，向甲方所在地人民法院提起诉讼",
        }
        r = client.post("/api/draft", json=payload, headers=auth)
        ok = r.status_code == 200
        check("起草-登录成功返回200", ok, f"status={r.status_code} {r.text[:120]}")
        d = r.json() if ok else {}
        check("起草-自动编号", d.get("contract_no") == no1, str(d.get("contract_no")))
        check("起草-正文含合同名称", payload["contract_name"] in d.get("draft_text", ""), "")
        check("起草-正文含金额大写", "肆拾捌万元整" in d.get("draft_text", ""), "")
        cid = d.get("contract", {}).get("id")

        r = client.post("/api/draft", json={**payload, "contract_type": "不存在"}, headers=auth)
        check("起草-未知类型返回400", r.status_code == 400, f"status={r.status_code}")

        # ---- B3 台账含状态 ----
        r = client.get("/api/contracts")
        row = next((c for c in r.json() if c["id"] == cid), None)
        check("台账-默认状态为草拟", row is not None and row.get("status") == "草拟", str(row))

        # ---- B3 编号自增 ----
        r = client.get("/api/contracts/next_no", params={"contract_type": "采购"})
        check("编号-同类型自增为002", r.json()["contract_no"].endswith("-002"), r.json()["contract_no"])

        # ---- B3 状态机流转 ----
        r = client.post(f"/api/contracts/{cid}/status", json={"status": "审批"}, headers=auth)
        check("状态-草拟→审批合法", r.status_code == 200 and r.json().get("status") == "审批", f"status={r.status_code}")

        r = client.post(f"/api/contracts/{cid}/status", json={"status": "归档"}, headers=auth)
        check("状态-审批→归档非法返回400", r.status_code == 400, f"status={r.status_code}")

        r = client.post(f"/api/contracts/{cid}/status", json={"status": "用印"})
        check("状态-未登录返回401", r.status_code == 401, f"status={r.status_code}")

        r = client.post("/api/contracts/99999/status", json={"status": "审批"}, headers=auth)
        check("状态-不存在合同返回404", r.status_code == 404, f"status={r.status_code}")

        # ---- B3 台账看板 ----
        r = client.get("/api/dashboard")
        dash = r.json()
        check("看板-总数1", dash["total"]["count"] == 1, str(dash["total"]))
        check("看板-采购类型计数", any(t["name"] == "采购" and t["count"] == 1 for t in dash["by_type"]), str(dash["by_type"]))
        check("看板-审批状态计数", any(s["name"] == "审批" and s["count"] == 1 for s in dash["by_status"]), str(dash["by_status"]))

        # ---- B3 CSV 导出 ----
        r = client.get("/api/export")
        body = r.content.decode("utf-8-sig")
        check("导出-含BOM", r.content.startswith(b"\xef\xbb\xbf"), "")
        check("导出-含表头", "合同名称" in body and "状态" in body, body.splitlines()[0] if body else "")
        check("导出-含数据行", "测试服务器采购合同" in body and "审批" in body, "")

    print(f"\nB3/B4 结果：通过 {_PASS}/{_PASS + _FAIL}")
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
