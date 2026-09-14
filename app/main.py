"""FastAPI 入口：上传 PDF → 分流 → 提取 → 入库 → 回显（考核 B1 全链路）。"""
import csv
import io
import logging
import os
import secrets
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import agents, cms, config, finance, logging_config, ocr, pdf_loader, pipeline, review_agent, store, table, vector_store

logging_config.setup_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    logger.info("向量库现有范例数：%d", vector_store.count())
    yield


app = FastAPI(title="AI 合同系统（最小集）", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request, call_next):
    """请求级日志：记录每一次 API 调用的方法、路径、状态码与耗时。"""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("HTTP %s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


class ExtractResponse(BaseModel):
    contract: dict
    scene: str
    rag_examples: list[str]


class ReviewRequest(BaseModel):
    contract_id: int


class AgentsRequest(BaseModel):
    text: str


class LoginRequest(BaseModel):
    username: str
    password: str


class StatusRequest(BaseModel):
    status: str


class DraftRequest(BaseModel):
    contract_type: str
    contract_name: str
    party_a: str
    party_b: str
    amount: float
    sign_date: str
    term_start: str
    term_end: str
    payment_method: str
    breach_liability: str
    dispute_resolution: str
    amount_capital: str = ""


class FinancePlanRequest(BaseModel):
    contract_type: str = "其他"
    payment_method: str = ""
    amount: float = 0.0


# 登录鉴权（考核 B4）：内存 token，演示级
_ACTIVE_TOKENS: set[str] = set()


def _require_auth(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization.removeprefix("Bearer ")
    if token not in _ACTIVE_TOKENS:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return token


@app.get("/api/health")
def health():
    return {"status": "ok", "examples_in_store": vector_store.count()}


@app.post("/api/login")
def login(req: LoginRequest):
    if req.username == config.ADMIN_USER and req.password == config.ADMIN_PASSWORD:
        token = secrets.token_hex(16)
        _ACTIVE_TOKENS.add(token)
        logger.info("登录成功：%s", req.username)
        return {"token": token}
    logger.warning("登录失败：%s", req.username)
    raise HTTPException(status_code=401, detail="账号或密码错误")


def _parse_pdf(tmp_path: str) -> tuple[str, str]:
    """PDF → 纯文本（文字版直接抽，扫描件走 OCR + 表格），返回 (text, scene)。"""
    text, scene = pdf_loader.extract_text_from_pdf(tmp_path)
    logger.info("PDF 场景分流：%s（%d 字符）", scene, len(text))
    if scene == "scanned":
        text = ocr.ocr_pdf(tmp_path)
        # 复杂表格/骑缝章/歪斜：结构化提取，作为全文的结构化补充
        table_text = table.extract_tables_from_pdf(tmp_path)
        if table_text.strip():
            text = (text + "\n\n" + table_text).strip()
            scene = "scanned_table"
        elif text.strip():
            scene = "scanned_ocr"
    return text, scene


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_contract(file: UploadFile = File(...)):
    start = time.perf_counter()
    # 1. 落盘临时文件
    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    logger.info("收到上传：%s（%d 字节）", file.filename, len(content))

    try:
        # 2. PDF 场景分流（文字版直接抽文本，扫描件走 OCR + 表格管线）
        text, scene = _parse_pdf(tmp_path)
        if not text.strip():
            return ExtractResponse(contract={}, scene=scene, rag_examples=[])

        # 3. 线性流水线：分类 → RAG few-shot 提取（LangChain LCEL）
        outcome = pipeline.run(text)
        result = outcome["contract"]

        # 4. 入库
        contract_id = store.insert(result)
        logger.info("提取完成：type=%s 合同ID=%d 耗时=%.1fms", result.contract_type, contract_id, (time.perf_counter() - start) * 1000)

        return ExtractResponse(
            contract={
                **result.model_dump(),
                "id": contract_id,
                "type_confidence": outcome["confidence"],
            },
            scene=scene,
            rag_examples=result.reference_examples,
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/extract_contract")
async def extract_contract_full(file: UploadFile = File(...)):
    """Odoo 集成专用：上传 PDF → 解析 + 多智能体提取 + 审查，返回完整结果。"""
    start = time.perf_counter()
    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    logger.info("收到 PDF 上传（Odoo 集成）：%s（%d 字节）", file.filename, len(content))
    try:
        text, scene = _parse_pdf(tmp_path)
        if not text.strip():
            return {
                "scene": scene, "text": "", "contract": {},
                "verdict": "", "risks": [], "reference_examples": [], "trace": [],
            }
        result = agents.run_agents(text)
        logger.info(
            "Odoo 集成提取完成：type=%s 耗时=%.1fms",
            result.get("contract_type"), (time.perf_counter() - start) * 1000,
        )
        return {
            "scene": scene,
            "text": text,
            "contract": result.get("contract") or {},
            "verdict": result.get("verdict") or "",
            "risks": result.get("risks") or [],
            "reference_examples": result.get("reference_examples") or [],
            "trace": result.get("trace") or [],
        }
    finally:
        os.unlink(tmp_path)


@app.get("/api/contracts")
def contracts():
    return store.list_contracts()


@app.get("/api/search")
def search(q: str = ""):
    """检索可视化（考核 D1）：返回整份范例 + 条款切块两条检索结果（含相似度）。"""
    if not q.strip():
        return {"examples": [], "chunks": []}
    return {
        "examples": vector_store.search_similar(q),
        "chunks": vector_store.search_chunks(q),
    }


def _contract_summary(c: dict) -> str:
    """把已入库合同字段拼成供审查/业财使用的文本摘要。"""
    return (
        f"合同名称：{c.get('contract_name')}\n"
        f"合同编号：{c.get('contract_no')}\n"
        f"类型：{c.get('contract_type')}\n"
        f"甲方：{c.get('party_a')}\n"
        f"乙方：{c.get('party_b')}\n"
        f"金额：{c.get('amount')} 元（{c.get('amount_capital')}）\n"
        f"签订日期：{c.get('sign_date')}\n"
        f"履行期限：{c.get('term_start')} 至 {c.get('term_end')}\n"
        f"付款方式：{c.get('payment_method')}\n"
        f"违约责任：{c.get('breach_liability')}\n"
        f"争议解决：{c.get('dispute_resolution')}\n"
    )


@app.post("/api/review")
def review_contract(req: ReviewRequest):
    """LangGraph 审查 Agent：审查已入库合同，返回结论 + 回路轨迹。"""
    contract = store.get_contract(req.contract_id)
    if contract is None:
        return {"error": "合同不存在"}
    summary = _contract_summary(contract)
    return review_agent.review(summary)


@app.post("/api/agents")
def run_agents_pipeline(req: AgentsRequest):
    """多智能体接力（考核 D5）：分类→提取→校验→审查，出错自我修正重来。"""
    return agents.run_agents(req.text)


@app.post("/api/contracts/{contract_id}/finance")
def generate_finance(contract_id: int):
    """业财一体化：从付款方式提取分期节点，生成收付款计划并落库。"""
    contract = store.get_contract(contract_id)
    if contract is None:
        return {"error": "合同不存在"}
    plans = finance.generate_plan(
        contract_id,
        contract["contract_type"],
        contract["payment_method"],
        contract["amount"],
    )
    return {
        "contract_id": contract_id,
        "direction": finance.direction_for(contract["contract_type"]),
        "plans": plans,
    }


@app.get("/api/contracts/{contract_id}/finance")
def get_finance(contract_id: int):
    return {"contract_id": contract_id, "plans": store.list_plans(contract_id)}


@app.post("/api/finance_plan")
def finance_plan(req: FinancePlanRequest):
    """Odoo 集成专用：无状态生成收付款计划（AI 提取分期节点，不落库）。"""
    installments = finance.extract_installments(req.payment_method, req.amount)
    return {
        "direction": finance.direction_for(req.contract_type),
        "plans": installments,
    }


@app.get("/api/templates")
def templates():
    """模板 + 条款库（考核 B3：要素拼装起草）。"""
    return cms.list_templates()


@app.get("/api/contracts/next_no")
def next_no(contract_type: str = "其他"):
    return {"contract_no": store.next_contract_no(contract_type)}


@app.post("/api/draft", dependencies=[Depends(_require_auth)])
def draft_contract(req: DraftRequest):
    """要素拼装起草（考核 B3）：模板拼装 + 自动编号 + 入库。"""
    try:
        result = cms.draft(**req.model_dump())
        logger.info("起草完成：编号=%s 名称=%s", result["contract_no"], req.contract_name)
        return result
    except ValueError as e:
        logger.warning("起草失败：%s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/contracts/{contract_id}/status", dependencies=[Depends(_require_auth)])
def change_status(contract_id: int, req: StatusRequest):
    """合同状态流转（考核 B3）：草拟→审批→用印→归档→作废。"""
    try:
        updated = store.update_status(contract_id, req.status)
    except ValueError as e:
        logger.warning("状态流转被拦截：合同 %d -> %s（%s）", contract_id, req.status, e)
        raise HTTPException(status_code=400, detail=str(e))
    if updated is None:
        logger.warning("状态流转失败：合同 %d 不存在", contract_id)
        raise HTTPException(status_code=404, detail="合同不存在")
    logger.info("状态流转：合同 %d -> %s", contract_id, req.status)
    return updated


@app.get("/api/dashboard")
def dashboard():
    """台账看板（考核 B3）：类型/状态/金额统计。"""
    return store.dashboard()


@app.get("/api/export")
def export_csv():
    """台账导出 CSV（考核 B3）。"""
    rows = store.list_contracts()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "合同名称", "合同编号", "类型", "甲方", "乙方", "金额(元)", "签订日期", "状态"])
    for r in rows:
        writer.writerow([
            r.get("id"), r.get("contract_name"), r.get("contract_no"), r.get("contract_type"),
            r.get("party_a"), r.get("party_b"), r.get("amount"), r.get("sign_date"), r.get("status"),
        ])
    content = "\ufeff" + buf.getvalue()  # BOM 便于 Excel 识别 UTF-8
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=contracts.csv"},
    )


# 静态页（放最后注册，避免覆盖 /api 路由）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
