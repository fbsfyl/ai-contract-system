"""工具函数：金额数字转中文大写（用于「金额大写校验」）。"""

# 中文数字与单位
_DIGITS = "零壹贰叁肆伍陆柒捌玖"
_UNITS = ["", "拾", "佰", "仟"]
_BIG_UNITS = ["", "万", "亿", "兆"]


def num_to_capital(num: float) -> str:
    """把数字金额转为中文大写（支持到分）。"""
    if num < 0:
        return "负" + num_to_capital(-num)

    # 拆整数与小数
    integer = int(num)
    decimal = round((num - integer) * 100)

    if integer == 0 and decimal == 0:
        return "零元整"

    int_str = _int_to_capital(integer)
    result = int_str + "元"

    if decimal == 0:
        return result + "整"

    jiao = decimal // 10
    fen = decimal % 10
    if jiao:
        result += _DIGITS[jiao] + "角"
    if fen:
        result += _DIGITS[fen] + "分"
    return result


def _int_to_capital(num: int) -> str:
    """整数部分转中文大写（分组处理，支持亿/万级）。"""
    if num == 0:
        return "零"

    groups = []
    while num > 0:
        groups.append(num % 10000)
        num //= 10000

    parts = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            continue
        g_str = _group_to_capital(g)
        parts.append(g_str + _BIG_UNITS[idx])

    # 中间补零处理
    result = ""
    for p in parts:
        if result and not result.endswith("零") and p.startswith("零"):
            pass
        result += p
    return result


def _group_to_capital(g: int) -> str:
    """四位一组转中文，含内部补零。"""
    s = ""
    zero_pending = False
    for pos in range(3, -1, -1):
        d = (g // (10 ** pos)) % 10
        unit = _UNITS[pos]
        if d == 0:
            zero_pending = True
        else:
            if zero_pending and s:
                s += "零"
            s += _DIGITS[d] + unit
            zero_pending = False
    return s
