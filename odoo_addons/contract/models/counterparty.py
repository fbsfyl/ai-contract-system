"""相对方 / 签约主体（考核 C：自建 model）。"""
from odoo import fields, models


class ContractCounterparty(models.Model):
    _name = "contract.counterparty"
    _description = "相对方 / 签约主体"

    name = fields.Char(string="名称", required=True)
    role = fields.Selection(
        [("party_a", "甲方"), ("party_b", "乙方")],
        string="签约角色",
    )
    contact = fields.Char(string="联系人")
    phone = fields.Char(string="电话")
    email = fields.Char(string="邮箱")
    contract_ids = fields.One2many("contract.contract", "counterparty_id", string="关联合同")
