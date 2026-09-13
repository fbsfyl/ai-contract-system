"""向量库封装（Chroma，考核 D1/D2）。

用途：存放「类型正确、字段规范」的标准合同范例。
新合同进来时先做向量相似检索，把最像的范例捞出来当 few-shot 参照（把 RAG 当监督学习用）。
"""
import json
import logging

import chromadb
from chromadb.config import Settings

from app import config
from app import embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "contract_examples"
CHUNK_COLLECTION_NAME = "contract_chunks"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        # 匿名模式关闭遥测，避免离线环境卡住
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _get_chunk_collection():
    return _get_client().get_or_create_collection(
        name=CHUNK_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_examples(items: list[dict]) -> None:
    """批量写入范例。每个 item 需含 id / text / metadata(含标准字段 JSON)。"""
    if not items:
        return
    col = get_collection()
    ids = [it["id"] for it in items]
    docs = [it["text"] for it in items]
    metas = [it["metadata"] for it in items]
    vectors = embeddings.embed(docs)
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    logger.info("已写入 %d 条合同范例到向量库", len(items))


def search_similar(text: str, top_k: int | None = None) -> list[dict]:
    """相似检索：返回最像的 top_k 条范例（含标准字段答案与相似度）。"""
    top_k = top_k or config.TOP_K
    col = get_collection()
    if col.count() == 0:
        return []
    query_vec = embeddings.embed([text])
    result = col.query(query_embeddings=query_vec, n_results=top_k)
    out = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    for i, doc, meta, dist in zip(ids, docs, metas, distances):
        out.append(
            {
                "id": i,
                "text": doc,
                "contract_type": meta.get("contract_type", ""),
                "fields": json.loads(meta.get("fields", "{}")),
                # cosine 距离 → 相似度（1 - distance），便于检索可视化
                "similarity": round(1.0 - float(dist), 4),
            }
        )
    return out


def count() -> int:
    return get_collection().count()


def split_contract_by_clause(text: str) -> list[str]:
    """按条款/语义切块（考核 D1 的切块策略）。

    理由：合同格式化文本中「一行 ≈ 一个字段/一条独立条款」（如「合同编号：…」
    「付款方式：…」），按行切即为按语义切，而非无脑按 500 字硬切——这样能保证
    「合同金额」与「金额大写」等强关联字段不被拆散，检索命中更有解释性。
    """
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def index_chunks(contract_id: str, contract_type: str, text: str) -> int:
    """把合同按条款切块、向量化并入库到 chunk 集合，返回块数。"""
    chunks = split_contract_by_clause(text)
    if not chunks:
        return 0
    col = _get_chunk_collection()
    ids = [f"{contract_id}#{i}" for i in range(len(chunks))]
    metas = [
        {"contract_id": contract_id, "contract_type": contract_type, "chunk_index": i}
        for i in range(len(chunks))
    ]
    vectors = embeddings.embed(chunks)
    col.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=vectors)
    logger.info("合同 %s 切块入库 %d 块", contract_id, len(chunks))
    return len(chunks)


def search_chunks(text: str, top_k: int | None = None) -> list[dict]:
    """条款级检索：返回最相似的条款块（含相似度），用于检索可视化。"""
    top_k = top_k or config.TOP_K
    col = _get_chunk_collection()
    if col.count() == 0:
        return []
    query_vec = embeddings.embed([text])
    result = col.query(query_embeddings=query_vec, n_results=top_k)
    out = []
    for i, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0],
    ):
        out.append(
            {
                "id": i,
                "text": doc,
                "contract_id": meta.get("contract_id", ""),
                "contract_type": meta.get("contract_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "similarity": round(1.0 - float(dist), 4),
            }
        )
    return out


def chunk_count() -> int:
    return _get_chunk_collection().count()
