"""合同信息管理（考核 B3）：模板 + 条款库 + 要素拼装起草 + 金额大写。

模板与条款库见 data/templates.json；起草时按模板拼装正文、自动生成编号并入库。
"""
import json
import os

from app import config, store
from app.schemas import ContractFields, ExtractedContract

TEMPLATES_PATH = os.path.join(config.BASE_DIR, "data", "templates.json")

_DIGITS = "零壹贰叁肆伍陆柒捌玖"
_SECTION_UNITS = ["", "万", "亿", "万亿"]
_POS_UNITS = ["", "拾", "佰", "仟"]


def _four_digits_to_cn(n: int) -> str:
    """0 <= n < 10000 的四位数转中文（无节单位，如「肆拾捌」「捌仟零壹」）。"""
    if n == 0:
        return "零"
    s = ""
    zero = False
    pos = 0
    while n > 0:
        d = n % 10
        if d == 0:
            if s:
                zero = True
        else:
            if zero:
                s = "零" + s
                zero = False
            s = _DIGITS[d] + _POS_UNITS[pos] + s
        n //= 10
        pos += 1
    return s


def _int_to_cn(n: int) -> str:
    """非负整数转中文（按万/亿分节，正确处理节间零，如 480000→肆拾捌万）。"""
    if n == 0:
        return "零"
    result = ""
    zero = False
    idx = 0
    while n > 0:
        sec = n % 10000
        if sec == 0:
            if result:
                zero = True
        else:
            sec_str = _four_digits_to_cn(sec)
            if zero:
                result = sec_str + _SECTION_UNITS[idx] + "零" + result
                zero = False
            elif sec < 1000 and result:
                result = sec_str + _SECTION_UNITS[idx] + "零" + result
            else:
                result = sec_str + _SECTION_UNITS[idx] + result
        n //= 10000
        idx += 1
    return result


def amount_to_capital(amount) -> str:
    """人民币金额转中文大写（整数元 + 角分）。"""
    num = round(float(amount), 2)
    if num < 0:
        return "负" + amount_to_capital(-num)
    yuan = int(num)
    jiao = int(round(num * 10)) % 10
    fen = int(round(num * 100)) % 10

    s = _int_to_cn(yuan) + "元"

    if jiao == 0 and fen == 0:
        return s + "整"
    if jiao > 0:
        s += _DIGITS[jiao] + "角"
    if fen > 0:
        if jiao == 0:
            s += "零"
        s += _DIGITS[fen] + "分"
    return s


def list_templates() -> list[dict]:
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_template(contract_type: str) -> dict | None:
    for t in list_templates():
        if t["type"] == contract_type:
            return t
    return None


def draft(
    contract_type: str,
    contract_name: str,
    party_a: str,
    party_b: str,
    amount,
    sign_date: str,
    term_start: str,
    term_end: str,
    payment_method: str,
    breach_liability: str,
    dispute_resolution: str,
    amount_capital: str = "",
) -> dict:
    """要素拼装起草：按模板拼出合同正文 + 自动编号，并直接入库。"""
    tpl = get_template(contract_type)
    if tpl is None:
        raise ValueError(f"无该类型模板：{contract_type}")

    contract_no = store.next_contract_no(contract_type)
    amount_f = float(amount)
    capital = amount_capital or amount_to_capital(amount_f)

    text = tpl["body"].format(
        contract_name=contract_name,
        contract_no=contract_no,
        party_a=party_a,
        party_b=party_b,
        amount=amount_f,
        amount_capital=capital,
        sign_date=sign_date,
        term_start=term_start,
        term_end=term_end,
        payment_method=payment_method,
        breach_liability=breach_liability,
        dispute_resolution=dispute_resolution,
    )

    fields = ContractFields(
        contract_name=contract_name,
        contract_no=contract_no,
        contract_type=contract_type,
        party_a=party_a,
        party_b=party_b,
        amount=amount_f,
        amount_capital=capital,
        sign_date=sign_date,
        term_start=term_start,
        term_end=term_end,
        payment_method=payment_method,
        breach_liability=breach_liability,
        dispute_resolution=dispute_resolution,
    )
    contract = ExtractedContract(**fields.model_dump(), reference_examples=[])
    contract_id = store.insert(contract)

    return {
        "contract": {**fields.model_dump(), "id": contract_id},
        "draft_text": text,
        "contract_no": contract_no,
    }
