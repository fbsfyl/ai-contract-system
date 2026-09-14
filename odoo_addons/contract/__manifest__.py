{
    "name": "合同管理（AI 集成）",
    "version": "16.0.1.0.0",
    "category": "Sales",
    "summary": "承载合同业务；AI 提取/审查通过外部 FastAPI 服务解耦调用",
    "description": """
考核 C（框架选型）的 Odoo 自定义模块：用「模型驱动」承载合同业务，AI 能力与 Odoo 业务解耦。

- 自建 model：合同（contract.contract）+ 相对方/签约主体（contract.counterparty）
- 自定义 view + 菜单 + 权限组（合同管理员）
- 铁律：只写 addons，不改 Odoo 源码、不动业务原生能力
- AI 提取/审查走外部 FastAPI 服务（/api/agents），Odoo 只存业务与结果
""",
    "depends": ["base"],
    "data": [
        "security/contract_security.xml",
        "security/ir.model.access.csv",
        "views/counterparty_views.xml",
        "views/contract_views.xml",
        "views/menus.xml",
    ],
    "installable": True,
    "application": True,
    "license": "LGPL-3",
}
