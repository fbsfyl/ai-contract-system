"""FastAPI 入口：上传 PDF → 分流 → 提取 → 入库 → 回显（考核 B1 全链路）。"""
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import agents, finance, ocr, pdf_loader, pipeline, review_agent, store, table, vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    logger.info("向量库现有范例数：%d", vector_store.count())
    yield


app = FastAPI(title="AI 合同系统（最小集）", lifespan=lifespan)


class ExtractResponse(BaseModel):
    contract: dict
    scene: str
    rag_examples: list[str]


class ReviewRequest(BaseModel):
    contract_id: int


class AgentsRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {"status": "ok", "examples_in_store": vector_store.count()}


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_contract(file: UploadFile = File(...)):
    # 1. 落盘临时文件
    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2. PDF 场景分流（文字版直接抽文本，扫描件走 OCR + 表格管线）
        text, scene = pdf_loader.extract_text_from_pdf(tmp_path)
        if scene == "scanned":
            text = ocr.ocr_pdf(tmp_path)
            # 复杂表格/骑缝章/歪斜：结构化提取，作为全文的结构化补充
            table_text = table.extract_tables_from_pdf(tmp_path)
            if table_text.strip():
                text = (text + "\n\n" + table_text).strip()
                scene = "scanned_table"
            elif text.strip():
                scene = "scanned_ocr"
            else:
                return ExtractResponse(
                    contract={},
                    scene="scanned",
                    rag_examples=[],
                )

        # 3. 线性流水线：分类 → RAG few-shot 提取（LangChain LCEL）
        outcome = pipeline.run(text)
        result = outcome["contract"]

        # 4. 入库
        contract_id = store.insert(result)

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


# 静态页（放最后注册，避免覆盖 /api 路由）
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
