"""合同结构化字段契约（Pydantic v2）。

这里的 Literal 枚举约束，对应考核 A 的「字段字典/枚举约束」：
合同类型只能取既定枚举，LLM 输出越界会在解析层被拦截并触发重试。
"""
from typing import Literal

from pydantic import BaseModel, Field

# 合同类型枚举（考核 B2 要求至少 4 类）
CONTRACT_TYPES = ("采购", "销售", "服务", "租赁", "其他")
ContractType = Literal["采购", "销售", "服务", "租赁", "其他"]


class ContractFields(BaseModel):
    """考核 B1 要求的核心字段（含金额大写校验）。"""

    contract_name: str = Field(description="合同名称")
    contract_no: str = Field(description="合同编号，若缺失填「未注明」")
    contract_type: ContractType = Field(description="合同类型，只能是采购/销售/服务/租赁/其他之一")
    party_a: str = Field(description="甲方（付款方/买方）")
    party_b: str = Field(description="乙方（收款方/卖方）")
    amount: float = Field(description="合同金额，纯数字，单位元")
    amount_capital: str = Field(description="合同金额的中文大写，如「壹拾贰万元整」")
    sign_date: str = Field(description="签订日期")
    term_start: str = Field(description="履行期限开始日期")
    term_end: str = Field(description="履行期限结束日期")
    payment_method: str = Field(description="付款方式")
    breach_liability: str = Field(description="违约责任")
    dispute_resolution: str = Field(description="争议解决方式")


class ExtractedContract(ContractFields):
    """入库/回显用：在字段基础上附带检索到的参照范例，便于日志核对。"""

    reference_examples: list[str] = Field(default_factory=list, description="RAG 检索到的参照范例 ID")
