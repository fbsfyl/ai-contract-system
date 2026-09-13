"""合同入库与查询（SQLite，考核 B3 的 contract 台账最小集）。"""
import json
import sqlite3
from datetime import datetime

from app import config
from app.schemas import ExtractedContract

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_name TEXT,
    contract_no TEXT,
    contract_type TEXT,
    party_a TEXT,
    party_b TEXT,
    amount REAL,
    amount_capital TEXT,
    sign_date TEXT,
    term_start TEXT,
    term_end TEXT,
    payment_method TEXT,
    breach_liability TEXT,
    dispute_resolution TEXT,
    reference_examples TEXT,
    status TEXT DEFAULT '草拟',
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS payment_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER,
    direction TEXT,
    stage TEXT,
    ratio REAL,
    amount REAL,
    condition TEXT,
    created_at TEXT
)
"""

# 合同状态机（考核 B3）：草拟 → 审批 → 用印 → 归档 → 作废
STATUS_FLOW = {
    "草拟": ["审批", "作废"],
    "审批": ["用印", "作废"],
    "用印": ["归档", "作废"],
    "归档": ["作废"],
    "作废": [],
}

# 合同类型 → 编号前缀（配置域的编号规则）
_TYPE_PREFIX = {
    "采购": "CG",
    "销售": "XS",
    "服务": "FW",
    "租赁": "ZL",
    "其他": "QT",
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _conn()
    conn.executescript(_SCHEMA)
    # 旧库迁移：补 status 列（CREATE TABLE IF NOT EXISTS 不会改已有表结构）
    cols = [r[1] for r in conn.execute("PRAGMA table_info(contracts)").fetchall()]
    if "status" not in cols:
        conn.execute("ALTER TABLE contracts ADD COLUMN status TEXT DEFAULT '草拟'")
    conn.commit()
    conn.close()


def insert(contract: ExtractedContract) -> int:
    conn = _conn()
    cur = conn.execute(
        """
        INSERT INTO contracts (
            contract_name, contract_no, contract_type, party_a, party_b,
            amount, amount_capital, sign_date, term_start, term_end,
            payment_method, breach_liability, dispute_resolution,
            reference_examples, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contract.contract_name,
            contract.contract_no,
            contract.contract_type,
            contract.party_a,
            contract.party_b,
            contract.amount,
            contract.amount_capital,
            contract.sign_date,
            contract.term_start,
            contract.term_end,
            contract.payment_method,
            contract.breach_liability,
            contract.dispute_resolution,
            json.dumps(contract.reference_examples, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_contracts() -> list[dict]:
    conn = _conn()
    rows = conn.execute("SELECT * FROM contracts ORDER BY id DESC").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["reference_examples"] = json.loads(d.get("reference_examples") or "[]")
        out.append(d)
    return out


def get_contract(contract_id: int) -> dict | None:
    conn = _conn()
    row = conn.execute("SELECT * FROM contracts WHERE id = ?", (contract_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["reference_examples"] = json.loads(d.get("reference_examples") or "[]")
    return d


def insert_plan(contract_id: int, direction: str, stage: str, ratio: float, amount: float, condition: str) -> int:
    conn = _conn()
    cur = conn.execute(
        """
        INSERT INTO payment_plans (contract_id, direction, stage, ratio, amount, condition, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (contract_id, direction, stage, ratio, amount, condition, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_plans(contract_id: int) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT id, direction, stage, ratio, amount, condition FROM payment_plans WHERE contract_id = ? ORDER BY id ASC",
        (contract_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_plans(contract_id: int) -> None:
    conn = _conn()
    conn.execute("DELETE FROM payment_plans WHERE contract_id = ?", (contract_id,))
    conn.commit()
    conn.close()


def update_status(contract_id: int, new_status: str) -> dict | None:
    """状态流转（校验状态机），返回更新后的合同。"""
    contract = get_contract(contract_id)
    if contract is None:
        return None
    current = contract.get("status") or "草拟"
    if new_status not in STATUS_FLOW.get(current, []):
        raise ValueError(f"非法状态流转：{current} → {new_status}")
    conn = _conn()
    conn.execute("UPDATE contracts SET status = ? WHERE id = ?", (new_status, contract_id))
    conn.commit()
    conn.close()
    return get_contract(contract_id)


def next_contract_no(contract_type: str) -> str:
    """按「HT-类型前缀-年份-三位序号」自动生成下一个合同编号。"""
    prefix = _TYPE_PREFIX.get(contract_type, "QT")
    year = datetime.now().year
    like = f"HT-{prefix}-{year}-%"
    conn = _conn()
    rows = conn.execute(
        "SELECT contract_no FROM contracts WHERE contract_no LIKE ? ORDER BY contract_no DESC",
        (like,),
    ).fetchall()
    conn.close()
    max_seq = 0
    for r in rows:
        try:
            max_seq = max(max_seq, int(r["contract_no"].rsplit("-", 1)[-1]))
        except (ValueError, IndexError):
            continue
    return f"HT-{prefix}-{year}-{max_seq + 1:03d}"


def dashboard() -> dict:
    """台账看板统计：各类型数量/金额、各状态数量、总计。"""
    conn = _conn()
    by_type = [
        dict(r)
        for r in conn.execute(
            "SELECT contract_type AS name, COUNT(*) AS count, SUM(amount) AS amount "
            "FROM contracts GROUP BY contract_type"
        ).fetchall()
    ]
    by_status = [
        dict(r)
        for r in conn.execute(
            "SELECT status AS name, COUNT(*) AS count FROM contracts GROUP BY status"
        ).fetchall()
    ]
    total = conn.execute("SELECT COUNT(*) AS count, SUM(amount) AS amount FROM contracts").fetchone()
    conn.close()
    return {
        "by_type": by_type,
        "by_status": by_status,
        "total": {"count": total["count"] or 0, "amount": total["amount"] or 0},
    }
