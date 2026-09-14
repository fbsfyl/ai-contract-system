# AI 合同系统 — 运行与验收说明

一个「提示词工程 + RAG 底座 + 业务系统 + 编排评估」四层协同的 AI 合同系统最小集，覆盖合同上传、场景分流、结构化提取、分类、入库、业财一体化、合同审查全链路。

## 一、系统架构

```
┌─────────────────────────────────────────────────────┐
│  应用层：合同业务系统（FastAPI + SQLite）             │
│  上传 → 提取 → 入库 → 台账 → 业财计划 → 审查         │
├─────────────────────────────────────────────────────┤
│  编排层：LangChain 线性管道 · LangGraph 审查/多智能体接力 │
├─────────────────────────────────────────────────────┤
│  AI 底座：Embedding(bge) + Chroma 向量库 + RAG few-shot│
├─────────────────────────────────────────────────────┤
│  提示词工程：分层提示词 · JSON Schema · 枚举约束 · 重试│
├─────────────────────────────────────────────────────┤
│  输入层：PDF 场景分流（文字版 / 扫描件 OCR）          │
└─────────────────────────────────────────────────────┘
```

### 技术选型

| 层 | 技术 |
|---|---|
| LLM | DeepSeek（`deepseek-chat`，OpenAI 兼容协议） |
| Embedding | 本地 `BAAI/bge-small-zh-v1.5`（默认）或 OpenAI 兼容 API |
| 向量库 | Chroma（持久化） |
| 存储 | SQLite（`contracts.db`，含 contracts / payment_plans 两表） |
| OCR | RapidOCR（PaddleOCR 的 ONNX 运行时版，内置中文模型） |
| PDF 渲染 | pypdfium2 |
| 编排 | LangChain（线性流水线）+ LangGraph（审查/多智能体） |
| Web | FastAPI + 原生静态页 |

### 技术原理（考核 F：白板讲解要点）

**1. Transformer 与自注意力（分类/提取的 LLM 底座）**

- 合同文本 → 分词/子词（tokenizer）→ 每个 token 映射为向量（embedding）。
- 自注意力（Self-Attention）：每个 token 与全文其他 token 计算相关性权重（Q·Kᵀ / √d，经 softmax），再对 V 加权求和——这正是「合同金额」与「金额大写」能跨句对齐、分类能抓住「甲方是买方还是卖方」语义关系的机制。
- 多层堆叠 + 前馈网络，输出被解码为结构化 JSON（本系统用 `response_format=json_object` 约束）。

**2. Embedding 向量化（RAG 底座）**

- `bge-small-zh-v1.5` 把每个合同文本块映射为固定维度向量，训练目标使语义相近的文本在向量空间更近。
- 本系统把合同按「一行一条款」切块（见 `vector_store.split_contract_by_clause`），逐块向量化。

**3. 余弦相似度检索（把 RAG 当监督学习）**

- 新合同向量 q 与库中范例向量 v 的相似度 = cos(q, v) = (q·v) / (‖q‖‖v‖)，Chroma 用 cosine 距离（1 − 相似度）。
- 检索 top-k 相似范例作为 few-shot 注入提示词，让 LLM 参照正确字段口径抽取——不训练模型，用检索替代监督信号。

**4. 全链路白板推演**

```
上传 PDF → 场景分流(文字/扫描OCR) → 合同文本
  → 按条款切块 → Embedding 向量化 → 余弦相似检索 top-k 范例
  → few-shot 注入提示词 → LLM 结构化输出(JSON) → Schema 枚举校验
  → 失败重试兜底 → 入库(台账) → 业财计划 / 审查 Agent
```

## 二、目录结构

```
contract project/
├── app/
│   ├── main.py          # FastAPI 入口 + 全部 API
│   ├── config.py        # 配置（.env 驱动）
│   ├── logging_config.py# 统一日志（控制台 + 滚动文件 logs/app.log）
│   ├── schemas.py       # 字段契约（Pydantic + 类型枚举）
│   ├── prompts.py       # 提示词工程（分类/提取分层）
│   ├── llm.py           # LLM 客户端（JSON 输出 + 解析重试）
│   ├── embeddings.py    # Embedding（本地 / API 双通道）
│   ├── vector_store.py  # Chroma 向量库
│   ├── pdf_loader.py    # PDF 文字/扫描分流
│   ├── ocr.py           # 扫描件 OCR 管线
│   ├── table.py         # 复杂表格/图像预处理管线（PP-Structure 最小集）
│   ├── extractor.py     # 提取流水线（RAG few-shot + 校验重试）
│   ├── classifier.py    # 合同分类（LLM + 规则双通道，输出置信度）
│   ├── cms.py           # 合同信息管理（模板+条款库起草+金额大写+自动编号）
│   ├── pipeline.py      # 线性流水线（LangChain LCEL：分类 → 提取）
│   ├── agents.py        # 多智能体接力（LangGraph：分类→提取→校验→审查，自我修正）
│   ├── finance.py       # 业财一体化（付款节点 → 收付款计划）
│   ├── review_agent.py  # LangGraph 审查 Agent
│   ├── store.py         # SQLite 入库/查询 + 收付款计划 + 状态机 + 台账看板
│   └── static/index.html
├── data/
│   ├── examples/contracts.json  # 10 条标准范例（RAG 检索库）
│   ├── tests.json               # 10 条测试金标准（采购/销售/服务/租赁/其他 各 2 条）
│   ├── templates.json           # 5 类合同模板 + 条款库（B3 起草）
│   ├── sample_contract.pdf      # 政府采购示范文本（空白模板）
│   ├── filled_contract.pdf      # 已填好的服务器采购合同
│   ├── scanned_contract.pdf     # 扫描版（图片型 PDF，测 OCR）
│   └── table_contract.pdf       # 含表格的合同（测表格结构化提取）
├── scripts/
│   ├── seed_examples.py         # 范例写入向量库
│   ├── evaluate.py              # few-shot 对比评估
│   ├── eval_classify.py         # 分类准确率评估（B2）
│   ├── eval_ocr.py              # 文字版 vs 扫描版准确率对比
│   ├── ragas_eval.py            # RAGAS 评估（4 指标 + 评估报告）
│   ├── test_unit.py             # 核心模块单元测试（unittest）
│   ├── test_agents.py           # 验证多智能体接力（D5）
│   ├── generate_sample_pdf.py   # 生成文字版测试 PDF
│   ├── generate_scanned_pdf.py  # 生成扫描版测试 PDF
│   └── test_table.py            # 验证复杂表格管线
├── odoo_addons/
│   └── contract/                # Odoo 自定义模块（考核 C：框架选型 5 分档）
│       ├── __manifest__.py
│       ├── models/
│       │   ├── contract.py      # 合同 model（13 字段 + 状态机 + AI 客户端）
│       │   ├── counterparty.py  # 相对方/签约主体 model
│       │   └── payment_plan.py  # 收付款计划 model（业财一体化）
│       ├── views/
│       │   ├── contract_views.xml
│       │   ├── counterparty_views.xml
│       │   └── menus.xml
│       └── security/
│           ├── contract_security.xml  # 权限组
│           └── ir.model.access.csv
├── Dockerfile
├── docker-compose.yml           # FastAPI 应用容器化
├── docker-compose.odoo.yml      # Odoo 16 + PostgreSQL 一键启动
├── start.ps1                    # 一键启动 FastAPI（前台）
├── start_all.ps1                # 一键启动全部服务（后台）
├── stop_all.ps1                 # 一键关闭全部服务
├── run_tests.py                 # 一键跑全部测试用例
├── .gitignore
├── .dockerignore
├── requirements.txt
└── .env.example
```

## 三、环境准备

- Python 3.12（`sentence-transformers` / `chromadb` / `rapidocr` 均支持）
- 可访问 DeepSeek API（需 `DEEPSEEK_API_KEY`）
- 首次运行需联网下载本地 Embedding 模型 `bge-small-zh-v1.5`（约 130MB，之后缓存）

## 四、安装与运行

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 安装依赖（会拉取 torch、chromadb、opencv 等，体积较大）
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. 配置环境变量
copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxxx

# 4. 写入向量库范例（RAG 检索库，只需执行一次）
.venv\Scripts\python.exe scripts\seed_examples.py

# 5. 启动服务
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动后访问 http://127.0.0.1:8000 即可在浏览器上传 PDF 测试。

### 一键启动（推荐）

```powershell
.\start.ps1        # 仅启动 AI 合同系统（FastAPI），前台运行
.\start_all.ps1    # 一键启动全部服务（FastAPI + Odoo 16 + PostgreSQL），后台运行
.\stop_all.ps1     # 一键关闭全部服务
```

`start_all.ps1` 会后台启动 FastAPI（监听 `0.0.0.0:8000` 供 Odoo 容器访问）和 Odoo/PostgreSQL 容器，关窗口不影响。启动完成后可访问：

- FastAPI：http://127.0.0.1:8000
- Odoo：http://localhost:8069 （admin / admin）

查看实时日志（另开 PowerShell 窗口）：

```powershell
Get-Content logs\app.log -Wait          # 业务 + HTTP 请求日志（主日志，UTF-8 带 BOM）
Get-Content logs\fastapi_error.log -Wait  # uvicorn 进程 stderr（启动报错/崩溃时看）
```

关闭全部服务：`.\stop_all.ps1`

### Docker 化启动

```powershell
docker compose up --build   # 需先准备 .env（DEEPSEEK_API_KEY）
```

### 一键跑全部测试用例

```powershell
.venv\Scripts\python.exe -B run_tests.py        # 核心准确率/验证脚本
.venv\Scripts\python.exe -B run_tests.py --all  # 额外跑 RAGAS 评估
```

> 注意：若在受限沙箱/只读环境运行，Python 写字节码缓存到全局目录可能被拦截，此时加 `-B` 参数（如 `.venv\Scripts\python.exe -B scripts\seed_examples.py`）。普通本机环境无需 `-B`。

### 日志

应用日志同时输出到控制台与 `logs/app.log`（滚动文件，单文件 5MB × 最多 5 个历史）。每次 API 请求（方法/路径/状态码/耗时）、PDF 场景分流、分类与提取流水线、起草、状态流转、业财计划等内部流程都会被记录，便于排障与验收演示。第三方库（chromadb / httpx / uvicorn 等）日志已压到 WARNING 降噪。

## 五、功能验收清单

### 对照评分标准的验收项

| 维度 | 功能 | 验收方式 | 预期结果 |
|---|---|---|---|
| B1 变量提取 | 上传 PDF → 提取 13 字段 | 浏览器上传 `filled_contract.pdf` | 名称/甲乙双方/金额/大写/日期/付款方式/违约/争议解决全部正确 |
| B2 分类 | 合同类型判定（LLM+规则双通道） | 提取结果中「合同类型」字段；或运行 `scripts\eval_classify.py` | 采购/销售/服务/租赁/其他 之一；10 份样本分类准确率 ≥90%（实测 100%） |
| B3 信息管理 | 合同台账 + 状态流转 | 「已入库合同台账」卡片 | 显示所有合同（含状态列），支持 草拟→审批→用印→归档→作废 流转，非法流转被拦截 |
| B3 信息管理 | 模板+条款库起草 | 「要素拼装起草」卡片 | 按 5 类模板选型，填要素后自动编号（`HT-类型-年份-序号`）+ 拼装正文 + 入库 |
| B3 信息管理 | 台账看板 + 导出 | 「台账看板」卡片 | 按类型/状态/金额汇总，支持一键导出 CSV |
| B4 商用加固 | 登录鉴权 | 页面顶部登录栏；或直接调 `POST /api/draft` 等 | 未登录访问写操作接口返回 401，登录后携带 Bearer token 才可起草/改状态 |
| B5 业财一体化 | 收付款计划 | 台账点「业财」按钮 | 按合同类型判定应收/应付，生成分期计划 |
| C 技术解耦 | LLM/Embedding 可替换 | 改 `.env` 的 `LLM_BASE_URL` / `EMBEDDING_PROVIDER` | 无需改业务代码即可切换底座 |
| C 框架选型 | Odoo 自定义模块（高级解） | 安装 `odoo_addons/contract` 模块 | 自建 model（合同/相对方）+ view + 权限组，AI 服务解耦（见下方「Odoo 自定义模块」） |
| D2 RAG 当监督学习 | few-shot 提升准确率 | 运行 `scripts\evaluate.py` | 有 few-shot 准确率 > 无 few-shot |
| D1 检索可视化 | 整份范例 + 条款切块检索 | 首页「检索可视化」卡片或 `GET /api/search?q=...` | 返回相似度排序的范例与条款块 |
| D4 改进闭环 | RAGAS 4 指标 + 前后对比 | 运行 `scripts\ragas_eval.py` | 输出 4 指标，并对比「整份范例」vs「条款切块」两轮聚合分，写 `data/ragas_report.txt` |
| D3 LangChain 线性 | 分类→提取串行管道 | 上传 PDF 时 `app/pipeline.py` 自动走 LCEL | `RunnableLambda | RunnableLambda` 无状态一遍跑完 |
| D3 LangGraph 审查 | 合同审查 + 回路 | 台账点「审查」按钮 | 返回结论 + 回路轨迹（pass/supplement/人工介入） |
| E 场景分流 | 文字版直抽、扫描件 OCR | 分别上传 `filled_contract.pdf` 和 `scanned_contract.pdf` | 前者「文字版」，后者「扫描件(OCR识别)」且均能提取 |
| E 复杂表格 | 表格结构化提取 | 运行 `scripts\test_table.py` | 识别 4×4 表格，逐格 OCR 还原（含倾斜校正/去噪/二值化） |
| E 对比数据 | 文字版 vs 扫描版准确率 | 运行 `scripts\eval_ocr.py` | 同一金标准上两方案逐字段准确率（当前均 100%） |
| A 提示词工程 | 分层提示词 + 枚举 + 重试 | 代码见 `app/prompts.py`、`app/extractor.py` | 分类/提取分层，`Literal` 枚举约束，解析失败自动重试 |
| G 工程化交付 | Docker 化 + 一键启动 + 测试用例 | `docker compose up`、`.\start.ps1`、`run_tests.py` | 一键起服务；一键跑全部准确率/评估脚本 |

### Odoo 自定义模块（考核 C：框架选型 5 分档）

`odoo_addons/contract` 是一个纯 Odoo addon，用「模型驱动」承载合同业务，对应评分标准「用 Odoo 自定义模块 + 自建 model + view + 权限组，只写 addons 不改源码」的高级解。

**包含内容**

- `contract.contract`：合同 model，字段与 FastAPI 的 13 字段契约一致，含状态机（草拟→审批→用印→归档→作废）+ 自动编号（`HT-类型前缀-年份-序号`）
- `contract.counterparty`：相对方 / 签约主体 model（合同通过 `counterparty_id` 关联）
- `contract.payment_plan`：收付款计划 model（业财一体化，合同 1:N 计划）
- 自定义 view（form / tree）+ 菜单 + 权限组（合同管理员）
- AI 集成按钮（全部通过 HTTP 调外部 FastAPI，Odoo 不内嵌 AI）：
  - `从 PDF 导入`：上传 PDF → 调 `/api/extract_contract` → 解析 + 多智能体提取 + 审查 → 回填 13 字段 + 审查结论/风险点
  - `生成付款计划`：调 `/api/finance_plan` → LLM 提取分期节点 → 生成收付款计划
  - `检索相似范例`：调 `/api/search` → RAG 检索相似合同 + 相似度回填 + 弹窗展示

**三个解耦（附加理念分）**

1. 业务解耦：AI 提取/审查/业财/RAG 不在 Odoo 内实现，全部通过 HTTP 调外部 FastAPI 服务，Odoo 只存业务与结果
2. 技术解耦：LLM/Embedding 仍由 FastAPI 侧 `.env` 控制，可替换
3. 编排解耦：Agent 编排层（LangChain/LangGraph）独立在 `app/` 里，Odoo 无感知

**安装运行（Odoo 16）**

一键启动（推荐，Docker）：

```powershell
docker compose -f docker-compose.odoo.yml up -d
```

首次启动会自动拉取 `odoo:16` + `postgres:16` 镜像、启动数据库、安装「合同管理（AI 集成）」模块。启动完成后访问 http://localhost:8069 （默认账号 admin / admin）。

> 容器内 Odoo 调外部 AI 服务时，宿主机地址是 `host.docker.internal`（Windows/Mac）。`docker-compose.odoo.yml` 已自动注入 `AI_SERVICE_URL=http://host.docker.internal:8000`，无需手动配置；如需覆盖，可在「设置 → 技术 → 系统参数」配置 `contract.ai_service_url`。

手动安装（已有 Odoo 环境）：

```powershell
# 1. 把 addon 目录加入 Odoo 的 addons_path（或软链进 addons 目录）
# 2. Odoo「应用」里刷新本地模块 → 安装「合同管理（AI 集成）」
# 3. 先启动 AI 服务：.\start.ps1  （默认 http://127.0.0.1:8000）
# 4. （可选）配置 AI 服务地址的系统参数：
#    设置 → 技术 → 系统参数 → 新增 contract.ai_service_url = http://127.0.0.1:8000
```

在合同表单上传「合同 PDF」→ 点「从 PDF 导入」按钮，即调用外部 AI 服务解析并回填 13 字段，同时带出审查结论与风险点；点「生成付款计划」生成业财分期节点；点「检索相似范例」做 RAG 检索。

### 已知评估数据

`scripts/evaluate.py` 输出（10 条金标准，覆盖 5 类，13×10=130 字段）：

```
无 few-shot：98.5%  (128/130)
有 few-shot：100.0% (130/130)
提升：+1.5%  ↑ RAG few-shot 有效
```

`scripts/eval_classify.py` 输出（20 份样本：10 范例 + 10 测试金标准）：

```
规则通道：19/20 = 95.0%
LLM 通道：20/20 = 100.0%
（B2 要求 ≥90%，达标）
```

`scripts/eval_ocr.py` 输出（同一金标准，13 字段）：

```
文字版（pdfplumber 直抽）：100.0%  (13/13)
扫描版（RapidOCR 识别）：100.0%  (13/13)
差距：+0.0%
```

`scripts/ragas_eval.py` 输出（D4 改进闭环：两轮评估对比，均值写入 `data/ragas_report.txt`）：

```
第一轮（context = 整份范例全文）：
  faithfulness / answer_relevancy / context_precision / context_recall
第二轮（context = 条款切块检索结果）：
  同上 4 指标，并与第一轮逐项对比 ↑/↓ 变化量
```

> RAGAS 分数偏低是设计口径所致：answer 为完整字段 JSON、context 为参照范例，逐字段准确率仍以 `evaluate.py` 为准；报告内已写清诊断与改进建议。两轮对比用于验证「条款切块检索」对上下文精度的改进效果。

## 六、API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 + 向量库范例数 |
| POST | `/api/extract` | 上传 PDF（multipart），返回提取结果 + 场景 + RAG 范例 |
| POST | `/api/extract_contract` | 上传 PDF → 解析 + 多智能体提取 + 审查，供 Odoo 集成调用 |
| GET | `/api/contracts` | 查询所有已入库合同 |
| GET | `/api/search?q=...` | 检索可视化（整份范例 + 条款切块，含相似度） |
| POST | `/api/agents` | 多智能体接力（body `{"text": "..."}`） |
| POST | `/api/review` | LangGraph 审查（body `{"contract_id": 1}`） |
| POST | `/api/contracts/{id}/finance` | 生成收付款计划 |
| GET | `/api/contracts/{id}/finance` | 查询收付款计划 |
| POST | `/api/finance_plan` | 无状态业财端点：收合同要素 → LLM 提取分期节点（供 Odoo 集成调用） |
| POST | `/api/login` | 登录（body `{"username","password"}`，返回 Bearer token） |
| GET | `/api/templates` | 合同模板 + 条款库列表 |
| GET | `/api/contracts/next_no?contract_type=` | 按类型取下一可用合同编号 |
| POST | `/api/draft` | 要素拼装起草（需登录，自动编号 + 拼正文 + 入库） |
| POST | `/api/contracts/{id}/status` | 状态流转（需登录，body `{"status":"审批"}`，校验状态机） |
| GET | `/api/dashboard` | 台账看板（按类型/状态/金额汇总） |
| GET | `/api/export` | 导出合同台账 CSV（含 UTF-8 BOM） |

`/api/extract` 返回的 `scene` 取值：`text`（文字版）、`scanned_ocr`（扫描件已 OCR）、`scanned_table`（扫描件含表格，已结构化）、`scanned`（OCR 也失败）。

## 七、测试数据说明

| 文件 | 用途 |
|---|---|
| `data/filled_contract.pdf` | 已填好的服务器采购合同（48 万），测完整提取 |
| `data/scanned_contract.pdf` | 图片型 PDF（模拟扫描件），测 OCR 管线 |
| `data/table_contract.pdf` | 含 4×4 付款表格的合同，测表格结构化提取 |
| `data/sample_contract.pdf` | 政府采购示范文本（空白模板），字段多为「未注明」属正常 |
| `data/examples/contracts.json` | 10 条标准范例（RAG 检索库，few-shot 参照） |
| `data/tests.json` | 10 条测试金标准（采购/销售/服务/租赁/其他 各 2 条），用于提取+分类评估 |

可重新生成测试 PDF：

```powershell
.venv\Scripts\python.exe scripts\generate_sample_pdf.py      # 生成 filled_contract.pdf
.venv\Scripts\python.exe scripts\generate_scanned_pdf.py     # 生成 scanned_contract.pdf
.venv\Scripts\python.exe scripts\test_table.py               # 生成 table_contract.pdf 并验证表格管线
```

## 八、注意事项

1. **扫描件识别依赖清晰度**：`ocr.py` 默认 `scale=2.0` 放大渲染提升识别率；复杂表格/骑缝章/歪斜场景走 `table.py`（灰度化 + 去噪 + 倾斜校正 + Otsu 二值化 + 表格线检测 + 逐格 OCR）。
2. **Embedding 默认走本地**（需 torch）；如改用 API，设 `EMBEDDING_PROVIDER=api` 并填入 `EMBEDDING_API_KEY`，可注释掉 requirements 里的 `sentence-transformers`。
3. **审查结论非确定性**：法务审查较严格，即使字段齐全也可能因「违约金计算周期」「质保期」等细节判 `supplement → 人工介入`，这是有状态回路的正常表现，非 bug。
4. **金额大写校验**：若文本未给大写，LLM 会按金额自动换算标准中文大写；不一致时以正文数字为准。
