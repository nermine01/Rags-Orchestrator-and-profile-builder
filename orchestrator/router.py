import os, glob, numpy as np
from sqlalchemy import select
from app.db import SessionLocal
from app.models import Agent
from app.rag.retriever import Retriever
from orchestrator.llm import Embeddings
from app.config import (
    RAG_MIN_SCORE,
    RAG_INDEX_ROOT,
    ROUTE_FALLBACK,
    ROUTE_FALLBACK_MIN_CENTROID,
    ROUTE_FALLBACK_MIN_RAG,
)
# orchestrator/router.py (or a utils file)
from app.config import PCM_FASTPATH_KEYWORDS, PCM_AGENT_NAME
from app.db import SessionLocal
from sqlalchemy import text as sqltext

def maybe_pcm_agent_id(question: str) -> int | None:
    q = (question or "").lower()
    if any(k in q for k in PCM_FASTPATH_KEYWORDS):
        with SessionLocal() as s:
            row = s.execute(
                "SELECT id FROM agents WHERE name = :n", 
                {"n": PCM_AGENT_NAME}
            ).fetchone()
            return row[0] if row else None
    return None

def indexed_agents():
    res = []
    for p in glob.glob(os.path.join(RAG_INDEX_ROOT, "agent*_index", "index.faiss")):
        d = os.path.basename(os.path.dirname(p))  # agent{ID}_index
        try:
            aid = int(d.replace("agent", "").replace("_index", ""))
            res.append(aid)
        except:
            pass
    return sorted(set(res))

class AgentRouter:
    def __init__(self, topk_agents=5):
        self.emb = Embeddings()
        self.topk_agents = topk_agents

    def _centroid_scores(self, qv: np.ndarray):
        with SessionLocal() as db:
            rows = db.execute(select(Agent.id, Agent.name, Agent.centroid)).all()
        out=[]
        for aid, name, blob in rows:
            if aid == 0 or not blob:
                continue
            v = np.frombuffer(blob, dtype=np.float32)
            if v.size == 0: 
                continue
            cos = float(np.dot(qv, v) / ((np.linalg.norm(qv)+1e-12)*(np.linalg.norm(v)+1e-12)))
            out.append((cos, aid, name))
        out.sort(reverse=True, key=lambda x: x[0])
        return out

    def _rag_probe(self, agent_id: int, qv: np.ndarray):
        try:
            hits = Retriever(agent_id).search(qv.reshape(1, -1), top_k=4)
        except FileNotFoundError:
            return 0.0, []
        top = max([s for s,_ in hits], default=0.0)
        return float(top), hits

    # ---------- strict: only agents >= RAG_MIN_SCORE ----------
    def choose_agent(self, text: str):
        # --- fastpath check ---
        aid = maybe_pcm_agent_id(text)
        if aid is not None:
            return aid, {
                "route": "pcm-fastpath",
                "why": "keywords matched",
            }
        qv = self.emb.encode([text])[0].astype(np.float32)
        best = (0.0, 0, {"route": "router", "rag_score": 0.0})
        for cos, aid, name in self._centroid_scores(qv)[:self.topk_agents]:
            top, hits = self._rag_probe(aid, qv)
            if top > best[0]:
                best = (top, aid, {"route":"router","why":"centroid_shortlist",
                                   "centroid":cos,"rag_score":top,"hits":hits})
        if best[0] >= RAG_MIN_SCORE:
            return best[1], best[2]
        return 0, {"route":"router","why":"none_above_threshold","rag_score":best[0]}

    # ---------- relaxed: pick *someone* even if < threshold ----------
    def choose_agent_relaxed(self, text: str):
        if ROUTE_FALLBACK == "off":
            return 0, {"route": "router_relaxed", "why": "off"}

        qv = self.emb.encode([text])[0].astype(np.float32)
        agents_idx = set(indexed_agents())

        if ROUTE_FALLBACK == "best_rag":
            best = (0.0, 0, None)
            for aid in agents_idx:
                if aid == 0: 
                    continue
                top, hits = self._rag_probe(aid, qv)
                if top > best[0]:
                    best = (top, aid, hits)
            if best[0] >= ROUTE_FALLBACK_MIN_RAG and best[1] != 0:
                return best[1], {"route": "router_relaxed", "why": "best_rag",
                                 "rag_score": best[0], "hits": best[2]}
            return 0, {"route": "router_relaxed", "why": "best_rag_no_candidate", "best_rag": best[0]}

        # centroid fallback
        cents = self._centroid_scores(qv)
        for cos, aid, name in cents:
            if aid in agents_idx and cos >= ROUTE_FALLBACK_MIN_CENTROID:
                top, hits = self._rag_probe(aid, qv)  # may be small
                return aid, {"route":"router_relaxed","why":"centroid",
                             "centroid":cos,"rag_score":top,"hits":hits}
        return 0, {"route":"router_relaxed","why":"centroid_no_candidate"}

