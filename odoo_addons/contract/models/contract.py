"""合同（考核 C：自建 model，承载合同业务，AI 能力走外部服务解耦调用）。"""
import logging
from datetime import datetime

from odoo import _, api, fields, models
from odoo.exceptions import UserError

logger = logging.getLogger(__name__)

# 与 app/store.py 保持一致的状态机（考核 B3）
STATUS_FLOW = {
    "草拟": ["审批", "作废"],
    "审批": ["用印", "作废"],
    "用印": ["归档", "作废"],
    "归档": ["作废"],
    "作废": [],
}

_STATUS_SELECTION = [
    ("草拟", "草拟"),
    ("审批", "审批"),
    ("用印", "用印"),
    ("归档", "归档"),
    ("作废", "作废"),
]

CONTRACT_TYPE_SELECTION = [
    ("采购", "采购"),
    ("销售", "销售"),
    ("服务", "服务"),
    ("租赁", "租赁"),
    ("其他", "其他"),
]


class ContractContract(models.Model):
    _name = "contract.contract"
    _description = "合同"
    _order = "id desc"
    _rec_name = "contract_name"

    # ===== 结构化字段（与 app/schemas.py 的 ContractFields 对齐，考核 B1）=====
    contract_name = fields.Char(string="合同名称", required=True)
    contract_no = fields.Char(string="合同编号")
    contract_type = fields.Selection(CONTRACT_TYPE_SELECTION, string="合同类型")
    party_a = fields.Char(string="甲方")
    party_b = fields.Char(string="乙方")
    amount = fields.Float(string="合同金额（元）", digits=(16, 2))
    amount_capital = fields.Char(string="金额大写")
    # 日期保持字符串，与 AI 提取结果一致，避免日期格式解析失败
    sign_date = fields.Char(string="签订日期")
    term_start = fields.Char(string="履行开始日期")
    term_end = fields.Char(string="履行结束日期")
    payment_method = fields.Text(string="付款方式")
    breach_liability = fields.Text(string="违约责任")
    dispute_resolution = fields.Text(string="争议解决")

    # ===== 业务状态机（考核 B3）=====
    status = fields.Selection(
        _STATUS_SELECTION,
        string="状态",
        default="草拟",
        required=True,
    )

    # ===== 相对方关联 =====
    counterparty_id = fields.Many2one("contract.counterparty", string="相对方 / 签约主体")

    # ===== 业财一体化（考核 B5）：合同 → 收付款计划 =====
    payment_plan_ids = fields.One2many("contract.payment_plan", "contract_id", string="收付款计划")

    # ===== AI 集成（业务解耦：只保存原文与结果，AI 计算走外部 FastAPI 服务）=====
    pdf_file = fields.Binary(string="合同 PDF", attachment=True)
    ai_text = fields.Text(string="合同原文（供 AI 提取）")
    ai_verdict = fields.Char(string="审查结论")
    ai_risks = fields.Text(string="审查风险点")
    reference_examples = fields.Char(string="RAG 参照范例")

    def _ai_service_url(self):
        """AI 服务地址（业务解耦：Odoo 不内嵌 AI，只通过 HTTP 调外部服务）。"""
        return (
            self.env["ir.config_parameter"].sudo().get_param("contract.ai_service_url")
            or "http://127.0.0.1:8000"
        )

    def _check_transition(self, new_status):
        self.ensure_one()
        current = self.status or "草拟"
        if new_status not in STATUS_FLOW.get(current, []):
            raise UserError(_("非法状态流转：%s → %s") % (current, new_status))

    def action_advance(self):
        """推进到下一状态（草拟→审批→用印→归档）。"""
        self.ensure_one()
        _next_map = {"草拟": "审批", "审批": "用印", "用印": "归档"}
        nxt = _next_map.get(self.status)
        if not nxt:
            raise UserError(_("当前状态「%s」无法继续推进") % self.status)
        self._check_transition(nxt)
        self.status = nxt

    def action_cancel(self):
        """作废合同。"""
        self.ensure_one()
        self._check_transition("作废")
        self.status = "作废"

    def _apply_ai_result(self, contract, verdict, risks, reference_examples, text=None):
        """把外部 AI 服务返回的结构化结果回填到当前合同。"""
        values = {
            "contract_name": contract.get("contract_name") or self.contract_name,
            "contract_no": contract.get("contract_no") or self.contract_no,
            "contract_type": contract.get("contract_type") or self.contract_type,
            "party_a": contract.get("party_a") or self.party_a,
            "party_b": contract.get("party_b") or self.party_b,
            "amount": contract.get("amount") or 0.0,
            "amount_capital": contract.get("amount_capital") or self.amount_capital,
            "sign_date": contract.get("sign_date") or self.sign_date,
            "term_start": contract.get("term_start") or self.term_start,
            "term_end": contract.get("term_end") or self.term_end,
            "payment_method": contract.get("payment_method") or self.payment_method,
            "breach_liability": contract.get("breach_liability") or self.breach_liability,
            "dispute_resolution": contract.get("dispute_resolution") or self.dispute_resolution,
            "ai_verdict": verdict or "",
            "ai_risks": "\n".join(risks or []),
            "reference_examples": ", ".join(reference_examples or []),
        }
        if text is not None:
            values["ai_text"] = text
        self.write(values)

    def _reload_action(self):
        return {
            "type": "ir.actions.act_window",
            "res_model": "contract.contract",
            "res_id": self.id,
            "view_mode": "form",
            "target": "current",
        }

    def action_ai_extract(self):
        """对已填写的合同原文跑多智能体提取 + 审查（考核 D5）。"""
        self.ensure_one()
        if not self.ai_text or not self.ai_text.strip():
            raise UserError(_("请先上传 PDF 导入合同原文，或填写「合同原文」后再执行 AI 提取"))

        import requests

        url = self._ai_service_url().rstrip("/") + "/api/agents"
        try:
            resp = requests.post(url, json={"text": self.ai_text}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.exception("调用 AI 服务失败")
            raise UserError(_("调用 AI 服务失败（%s）：%s") % (url, exc)) from exc

        contract = data.get("contract") or {}
        self._apply_ai_result(
            contract,
            data.get("verdict") or "",
            data.get("risks") or [],
            data.get("reference_examples") or [],
        )
        return self._reload_action()

    def action_import_pdf(self):
        """上传 PDF → 外部 AI 服务解析 + 多智能体提取 + 审查 → 回填字段（核心整合）。"""
        self.ensure_one()
        if not self.pdf_file:
            raise UserError(_("请先上传「合同 PDF」再执行导入"))

        import base64

        import requests

        url = self._ai_service_url().rstrip("/") + "/api/extract_contract"
        pdf_bytes = base64.b64decode(self.pdf_file)
        filename = (self.contract_name or "contract").replace("/", "_") + ".pdf"
        try:
            resp = requests.post(
                url,
                files={"file": (filename, pdf_bytes, "application/pdf")},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.exception("调用 AI 服务失败")
            raise UserError(_("调用 AI 服务失败（%s）：%s") % (url, exc)) from exc

        contract = data.get("contract") or {}
        self._apply_ai_result(
            contract,
            data.get("verdict") or "",
            data.get("risks") or [],
            data.get("reference_examples") or [],
            text=data.get("text") or "",
        )
        return self._reload_action()

    def action_generate_finance(self):
        """业财一体化：调外部 AI 服务提取分期收付款节点，写入收付款计划。"""
        self.ensure_one()
        if not self.amount or self.amount <= 0:
            raise UserError(_("请先填写「合同金额」再生成收付款计划"))

        import requests

        url = self._ai_service_url().rstrip("/") + "/api/finance_plan"
        try:
            resp = requests.post(
                url,
                json={
                    "contract_type": self.contract_type or "其他",
                    "payment_method": self.payment_method or "",
                    "amount": self.amount,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.exception("调用 AI 服务失败")
            raise UserError(_("调用 AI 服务失败（%s）：%s") % (url, exc)) from exc

        direction = data.get("direction") or "应收"
        plans = data.get("plans") or []
        self.payment_plan_ids.unlink()
        for idx, it in enumerate(plans, start=1):
            self.env["contract.payment_plan"].create({
                "contract_id": self.id,
                "sequence": idx * 10,
                "direction": direction,
                "stage": it.get("stage") or "付款",
                "ratio": it.get("ratio") or 0.0,
                "amount": it.get("amount") or 0.0,
                "condition": it.get("condition") or "未注明",
            })
        return self._reload_action()

    def action_search_similar(self):
        """RAG 检索：调外部 AI 服务检索相似合同范例，回填参照范例字段。"""
        self.ensure_one()
        query = (self.ai_text or self.contract_name or "").strip()
        if not query:
            raise UserError(_("请先上传 PDF 导入合同原文，或填写合同名称，再检索相似范例"))

        import requests

        url = self._ai_service_url().rstrip("/") + "/api/search"
        try:
            resp = requests.get(url, params={"q": query}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.exception("调用 AI 服务失败")
            raise UserError(_("调用 AI 服务失败（%s）：%s") % (url, exc)) from exc

        examples = data.get("examples") or []
        refs = [f"{e.get('id')}（相似度 {e.get('similarity', 0):.2f}）" for e in examples]
        self.reference_examples = ", ".join(refs) or "未找到相似范例"
        return self._reload_action()

    @api.model
    def create(self, vals):
        """新建合同时若未填编号，按「HT-类型前缀-年份-序号」自动生成。"""
        if not vals.get("contract_no"):
            vals["contract_no"] = self._next_contract_no(vals.get("contract_type") or "其他")
        return super().create(vals)

    def _next_contract_no(self, contract_type):
        prefix = {"采购": "CG", "销售": "XS", "服务": "FW", "租赁": "ZL", "其他": "QT"}.get(contract_type, "QT")
        year = datetime.now().year
        like = f"HT-{prefix}-{year}-%"
        last = self.search([("contract_no", "like", like)], order="contract_no desc", limit=1)
        max_seq = 0
        if last and last.contract_no:
            try:
                max_seq = int(last.contract_no.rsplit("-", 1)[-1])
            except (ValueError, IndexError):
                max_seq = 0
        return f"HT-{prefix}-{year}-{max_seq + 1:03d}"
