import httpx
from orchestrator.llm import Embeddings
from app.rag.retriever import Retriever
from app.config import N8N_WEBHOOK_URL

_emb = Embeddings()

def rag_search(agent_id: int, query: str, top_k=4):
    vec = _emb.encode([query])
    return Retriever(agent_id).search(vec, top_k=top_k)

def call_n8n(payload: dict):
    if not N8N_WEBHOOK_URL:
        return {"status":"skipped", "reason":"N8N_WEBHOOK_URL not set"}
    with httpx.Client(timeout=30) as c:
        r = c.post(N8N_WEBHOOK_URL, json=payload)
        try: data = r.json()
        except Exception: data = {"text": r.text}
    return {"status": r.status_code, "data": data}
