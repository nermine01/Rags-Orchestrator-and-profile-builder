import os
from fastapi import FastAPI, UploadFile, Form, Depends
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Agent, Document
import app.listeners
from orchestrator.president import President
from app.api_pcm import router as pcm_router

app = FastAPI(title="President Orchestrator")

app.include_router(pcm_router)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def ensure_president_agent():
    # Ensure a real row exists for id=0 (so uploads to agent_id=0 pass FK)
    with SessionLocal() as db:
        prez = db.get(Agent, 0)
        if not prez:
            # Insert agent 0 explicitly
            from sqlalchemy import text
            db.execute(text("INSERT INTO agents (id, name, description) VALUES (0, 'President', 'Central memory')"))
            db.commit()
ensure_president_agent()
from pydantic import BaseModel
from orchestrator.president import President

class ChatReq(BaseModel):
    text: str
    agent_id: int | None = None
    force_route: bool = False  # if true, skip President-first

@app.post("/chat/json")
def chat_json(req: ChatReq):
    pres = President(None if req.force_route else req.agent_id)
    return pres.handle(req.text)


@app.post("/agents")
def create_agent(name: str = Form(...), description: str = Form("")):
    with SessionLocal() as db:
        a = Agent(name=name, description=description)
        db.add(a); db.commit(); db.refresh(a)
        return {"id": a.id, "name": a.name}

@app.post("/agents/{agent_id}/documents")
def upload_document(agent_id: int, file: UploadFile, db: Session = Depends(get_db)):
    os.makedirs("./uploads", exist_ok=True)
    path = f"./uploads/agent{agent_id}_{file.filename}"
    with open(path, "wb") as f: f.write(file.file.read())
    d = Document(agent_id=agent_id, mime=file.content_type or "application/pdf", status=False, file_path=path)
    db.add(d); db.commit(); db.refresh(d)
    return {"doc_id": d.id, "status": d.status}

@app.post("/chat")
def chat(text: str = Form(...), agent_id: int | None = Form(None), user_id: str = Form("anonymous")):
    pres = President(agent_id)
    return pres.handle(text)
# In your venv, from the project root
# $env:PYTHONPATH = (Get-Location).Path
# python -m celery -A app.celery_app:celery_app worker -Q rag -l info -P solo --concurrency=1
import os, json, glob, numpy as np
from sqlalchemy import select
from app.db import SessionLocal
from app.models import Agent
from app.rag.ingestor import Ingestor
from orchestrator.llm import Embeddings
from orchestrator.router import AgentRouter
from app.rag.retriever import Retriever
from fastapi import Query
from app.config import RAG_MIN_SCORE

@app.get("/admin/agents/health")
def agents_health():
    rows=[]
    with SessionLocal() as db:
        agents = db.execute(select(Agent.id, Agent.name, Agent.centroid)).all()
    for aid, name, cent in agents:
        idx_dir = Ingestor.index_dir_for_agent(aid)
        faiss_ok = os.path.exists(os.path.join(idx_dir, "index.faiss"))
        meta = os.path.join(idx_dir, "meta.jsonl")
        chunks = 0
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                chunks = sum(1 for _ in f)
        rows.append({"agent_id":aid, "name":name, "index":faiss_ok, "chunks":chunks, "has_centroid":bool(cent)})
    return {"agents": rows}

@app.get("/admin/router/explain")
def router_explain(q: str = Query(..., description="Your query")):
    emb = Embeddings()
    r = AgentRouter()
    qv = emb.encode([q])[0].astype(np.float32)

    cents = []
    with SessionLocal() as db:
        rows = db.execute(select(Agent.id, Agent.name, Agent.centroid)).all()
    for aid, name, blob in rows:
        if not blob or aid == 0: continue
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size == 0: continue
        score = float(np.dot(qv, v) / ((np.linalg.norm(qv)+1e-12)*(np.linalg.norm(v)+1e-12)))
        cents.append({"agent_id": aid, "name": name, "centroid_cos": round(score, 3)})
    cents.sort(key=lambda x: x["centroid_cos"], reverse=True)

    probes = []
    for c in cents[:5]:
        hits = Retriever(c["agent_id"]).search(qv.reshape(1,-1), top_k=4)
        top_rag = max([s for s,_ in hits], default=0.0)
        probes.append({**c, "top_rag": round(float(top_rag), 3), "meets_threshold": top_rag >= RAG_MIN_SCORE, "hits": hits})
    return {"threshold": RAG_MIN_SCORE, "centroid_top": cents[:5], "probes": probes}
