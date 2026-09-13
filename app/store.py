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


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _conn()
    conn.executescript(_SCHEMA)
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
