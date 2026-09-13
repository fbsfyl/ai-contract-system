"""提示词工程层（考核 A）。

设计要点：
1. 分层：分类提示词 与 提取提示词 分离（层次③「分类/提取分层拆解」）。
2. 提取提示词内嵌 JSON Schema 约束 + 字段枚举约束 + few-shot 占位（层次②）。
3. 使用 JSON 结构化输出（response_format=json_object），配合解析重试兜底。
"""

# ---- 分类提示词 ----
CLASSIFY_SYSTEM_PROMPT = """你是资深合同审查员。请根据合同文本判断合同类型。

类型只能是以下枚举之一：采购、销售、服务、租赁、其他。
判定规则：
- 采购：甲方是买方/采购方，乙方是供应商，围绕货物采购。
- 销售：甲方是卖方/销售方，乙方是买方，围绕货物销售。
- 服务：以提供服务（技术、咨询、劳务等）为核心。
- 租赁：以租赁标的物（房屋、设备等）为核心。
- 其他：以上均不匹配。

只输出一个 JSON 对象，格式如下，不要输出任何多余文字：
{"contract_type": "采购", "confidence": 0.9}
confidence 为 0~1 的小数，表示你的把握程度。"""


# ---- 提取提示词（含 few-shot 占位与 Schema 约束）----
EXTRACT_SYSTEM_PROMPT = """你是资深合同信息抽取专家。请从给定合同文本中抽取结构化字段。

【字段枚举约束】contract_type 只能是以下之一：采购、销售、服务、租赁、其他。

【金额大写校验】amount 输出纯数字（单位元）；amount_capital 输出中文大写。
如果文本未给出大写，请根据 amount 自行换算成标准中文大写（如 120000 -> 壹拾贰万元整）。
若两处金额不一致，以正文数字为准并在 amount 中体现，amount_capital 填正确的大写。

【缺失字段处理】确无信息的字段填「未注明」。

【输出要求】只输出一个 JSON 对象，键名严格如下，不要输出任何多余文字、不要加 markdown 代码块：
{
  "contract_name": "",
  "contract_no": "",
  "contract_type": "",
  "party_a": "",
  "party_b": "",
  "amount": 0,
  "amount_capital": "",
  "sign_date": "",
  "term_start": "",
  "term_end": "",
  "payment_method": "",
  "breach_liability": "",
  "dispute_resolution": ""
}
"""

# few-shot 示例模板：由 extractor 在检索到相似范例后注入
FEWSHOT_TEMPLATE = """以下是与你任务相关的「标准合同范例」及已正确抽取的字段，供你参照（字段规范、类型口径以它们为准）：

{examples}

---

下面是待抽取的合同文本：
{text}
"""

ZERO_SHOT_TEMPLATE = """下面是待抽取的合同文本：
{text}
"""


def build_classify_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_extract_messages(text: str, few_shot: str = "") -> list[dict]:
    if few_shot:
        user_content = FEWSHOT_TEMPLATE.format(examples=few_shot, text=text)
    else:
        user_content = ZERO_SHOT_TEMPLATE.format(text=text)
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
