# -*- coding: utf-8 -*-
"""收付款计划（考核 B5 业财一体化）：合同 → 分期收付款节点。"""
from odoo import fields, models


class ContractPaymentPlan(models.Model):
    _name = "contract.payment_plan"
    _description = "收付款计划"
    _order = "sequence, id"

    sequence = fields.Integer(string="期次", default=10)
    contract_id = fields.Many2one(
        "contract.contract", string="合同", ondelete="cascade", required=True
    )
    direction = fields.Selection(
        [("应收", "应收"), ("应付", "应付")], string="收付方向"
    )
    stage = fields.Char(string="阶段")
    ratio = fields.Float(string="比例")
    amount = fields.Float(string="金额（元）", digits=(16, 2))
    condition = fields.Char(string="触发条件")
