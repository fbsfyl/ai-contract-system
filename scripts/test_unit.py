"""核心模块单元测试（不调 LLM/网络，纯逻辑，快）。

覆盖：
- classifier 规则通道关键词命中
- finance 方向映射 + 分期兜底
- extractor 字段归一化（金额解析 + 类型越界）
- schemas 枚举约束
- vector_store 条款切块
- evaluate 逐字段核对逻辑

用法：python -B scripts/test_unit.py
"""
import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from app import cms, classifier, config, extractor, finance, store, vector_store  # noqa: E402
from app.schemas import ContractFields, ExtractedContract  # noqa: E402
from evaluate import FIELDS, compare, field_equal  # noqa: E402

VALID_FIELDS = {
    "contract_name": "测试合同",
    "contract_no": "HT-0001",
    "contract_type": "采购",
    "party_a": "甲方公司",
    "party_b": "乙方公司",
    "amount": 100000,
    "amount_capital": "壹拾万元整",
    "sign_date": "2025年1月1日",
    "term_start": "2025年1月1日",
    "term_end": "2025年12月31日",
    "payment_method": "一次性付清",
    "breach_liability": "按日违约金",
    "dispute_resolution": "协商解决",
}


class TestClassifierRules(unittest.TestCase):
    def test_purchase_keywords(self):
        t, c = classifier.classify_by_rules("采购合同 买方 供应商")
        self.assertEqual(t, "采购")
        self.assertEqual(c, 0.9)

    def test_lease_keywords(self):
        t, _ = classifier.classify_by_rules("房屋租赁合同 租金 租期")
        self.assertEqual(t, "租赁")

    def test_service_keywords(self):
        t, _ = classifier.classify_by_rules("服务合同 委托 咨询")
        self.assertEqual(t, "服务")

    def test_sales_keywords(self):
        t, _ = classifier.classify_by_rules("销售合同 卖方 出售")
        self.assertEqual(t, "销售")

    def test_no_match(self):
        self.assertEqual(classifier.classify_by_rules("保密协议"), (None, 0.0))


class TestFinanceDirection(unittest.TestCase):
    def test_directions(self):
        self.assertEqual(finance.direction_for("采购"), "应付")
        self.assertEqual(finance.direction_for("服务"), "应付")
        self.assertEqual(finance.direction_for("销售"), "应收")
        self.assertEqual(finance.direction_for("租赁"), "应收")
        self.assertEqual(finance.direction_for("其他"), "应收")

    def test_unknown_default(self):
        self.assertEqual(finance.direction_for("未知类型"), "应收")


class TestFinanceNormalize(unittest.TestCase):
    def test_ratio_fills_amount(self):
        out = finance._normalize([{"stage": "预付款", "ratio": 0.5}], 100)
        self.assertEqual(out[0]["amount"], 50.0)

    def test_amount_backfills_ratio(self):
        out = finance._normalize([{"stage": "首款", "amount": 50}], 100)
        self.assertEqual(out[0]["ratio"], 0.5)

    def test_empty_fallback(self):
        out = finance._normalize([], 100)
        self.assertEqual(out, [{"stage": "全款", "ratio": 1.0, "amount": 100, "condition": "未注明"}])


class TestExtractorNormalize(unittest.TestCase):
    def test_amount_with_comma(self):
        data = dict(VALID_FIELDS, amount="100,000")
        self.assertEqual(extractor._normalize(data).amount, 100000.0)

    def test_invalid_amount(self):
        with self.assertRaises(ValueError):
            extractor._normalize(dict(VALID_FIELDS, amount="abc"))

    def test_invalid_contract_type(self):
        with self.assertRaises(ValueError):
            extractor._normalize(dict(VALID_FIELDS, contract_type="借贷"))


class TestSchemas(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(ContractFields(**VALID_FIELDS).contract_name, "测试合同")

    def test_invalid_type(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            ContractFields(**dict(VALID_FIELDS, contract_type="借贷"))


class TestSplitClause(unittest.TestCase):
    def test_split_lines(self):
        chunks = vector_store.split_contract_by_clause("第一行\n\n第二行\n  \n第三行")
        self.assertEqual(chunks, ["第一行", "第二行", "第三行"])


class TestFieldCompare(unittest.TestCase):
    def test_amount_equal(self):
        self.assertTrue(field_equal("amount", 480000, 480000))
        self.assertFalse(field_equal("amount", 480000, 480001))

    def test_date_equal(self):
        self.assertTrue(field_equal("sign_date", "2025年2月20日", "2025年2月20日"))
        self.assertFalse(field_equal("sign_date", "2025年2月20日", "2025年3月8日"))

    def test_string_normalized(self):
        self.assertTrue(field_equal("contract_name", "服务器采购合同", "服务器 采购 合同"))
        self.assertFalse(field_equal("contract_name", "服务器采购合同", "服务器采购协议"))

    def test_compare_all(self):
        correct, total = compare(VALID_FIELDS, VALID_FIELDS)
        self.assertEqual((correct, total), (len(FIELDS), len(FIELDS)))


class TestStore(unittest.TestCase):
    def setUp(self):
        self._orig_db = config.DB_PATH
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        self._tmp_path = tmp.name
        config.DB_PATH = self._tmp_path
        store.init_db()

    def tearDown(self):
        config.DB_PATH = self._orig_db
        if os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)

    def _insert(self, contract_type="采购", contract_no="HT-CG-2026-001"):
        c = ContractFields(
            contract_name="测试合同", contract_no=contract_no, contract_type=contract_type,
            party_a="甲方", party_b="乙方", amount=100000, amount_capital="壹拾万元整",
            sign_date="2026年1月1日", term_start="2026年1月1日", term_end="2026年12月31日",
            payment_method="一次性付清", breach_liability="违约", dispute_resolution="协商",
        )
        return store.insert(ExtractedContract(**c.model_dump(), reference_examples=[]))

    def test_default_status(self):
        cid = self._insert()
        self.assertEqual(store.get_contract(cid)["status"], "草拟")

    def test_valid_transition(self):
        cid = self._insert()
        self.assertEqual(store.update_status(cid, "审批")["status"], "审批")

    def test_invalid_transition(self):
        cid = self._insert()
        with self.assertRaises(ValueError):
            store.update_status(cid, "归档")  # 草拟不能直接归档

    def test_next_no(self):
        from datetime import datetime

        year = datetime.now().year
        self.assertEqual(store.next_contract_no("采购"), f"HT-CG-{year}-001")

    def test_dashboard_empty(self):
        self.assertEqual(store.dashboard()["total"]["count"], 0)


class TestCms(unittest.TestCase):
    def test_amount_capital(self):
        self.assertEqual(cms.amount_to_capital(480000), "肆拾捌万元整")
        self.assertEqual(cms.amount_to_capital(0), "零元整")

    def test_templates_five_types(self):
        types = {t["type"] for t in cms.list_templates()}
        self.assertEqual(types, {"采购", "销售", "服务", "租赁", "其他"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
