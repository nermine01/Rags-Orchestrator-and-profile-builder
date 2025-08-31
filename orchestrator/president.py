from orchestrator.llm import LLM, Embeddings
from orchestrator.router import AgentRouter
from app.rag.retriever import Retriever
from app.config import RAG_MIN_SCORE, ROUTE_FALLBACK
from app.db import SessionLocal
from app.models import Agent

class President:
    def __init__(self, agent_id: int | None = None):
        self.agent_id = agent_id
        self.llm = LLM()
        self.router = AgentRouter()
        self.emb = Embeddings()

    def _agent_name(self, aid: int) -> str:
        if aid == 0: return "President"
        with SessionLocal() as db:
            a = db.get(Agent, aid)
            return a.name if a and a.name else f"Agent {aid}"

    def _rag(self, agent_id: int, user_text: str):
        qv = self.emb.encode([user_text])
        hits = Retriever(agent_id).search(qv, top_k=4)
        top = max([s for s,_ in hits], default=0.0)
        return top, hits

    def _answer(self, agent_id: int, user_text: str, hits):
        name = self._agent_name(agent_id)
        ctx = "".join([f"\n[doc:{m.get('doc_id')} chunk:{m.get('chunk')} score:{s:.3f}]" for s, m in hits])
        prompt = (
            f"You are {name}.\n"
            f"User: {user_text}\n"
            f"Context:{ctx or ' (none) '}\n"
            f"Answer briefly. Ground in context if available; if not, say you are answering without context."
        )
        return self.llm.complete(prompt)

    def handle(self, user_text: str) -> dict:
        # If caller forced an agent: try that first strictly, then route.
        if self.agent_id is not None:
            top, hits = self._rag(self.agent_id, user_text)
            if top >= RAG_MIN_SCORE:
                return {"type":"answer","chosen_agent":self.agent_id,"text":self._answer(self.agent_id,user_text,hits),"used_chunks":hits}

            chosen, meta = self.router.choose_agent(user_text)  # strict
            if chosen != 0 and meta.get("rag_score", 0.0) >= RAG_MIN_SCORE:
                h = meta.get("hits") or self._rag(chosen, user_text)[1]
                return {"type":"answer","chosen_agent":chosen,"meta":meta,"text":self._answer(chosen,user_text,h),"used_chunks":h}

            if ROUTE_FALLBACK != "off":  # relaxed
                rc, rmeta = self.router.choose_agent_relaxed(user_text)
                if rc != 0:
                    h = rmeta.get("hits") or []
                    return {"type":"answer","chosen_agent":rc,"meta":rmeta,"text":self._answer(rc,user_text,h),"used_chunks":h}

            # Final: low confidence via requested agent
            return {"type":"answer","chosen_agent":self.agent_id,"meta":{"warning":"low-context"},"text":self._answer(self.agent_id,user_text,[]),"used_chunks":[]}

        # No agent specified → President first (strict)
        top0, hits0 = self._rag(0, user_text)
        if top0 >= RAG_MIN_SCORE:
            return {"type":"answer","chosen_agent":0,"meta":{"route":"president","rag_top":top0},"text":self._answer(0,user_text,hits0),"used_chunks":hits0}

        # Route strictly
        chosen, meta = self.router.choose_agent(user_text)
        if chosen != 0 and meta.get("rag_score", 0.0) >= RAG_MIN_SCORE:
            h = meta.get("hits") or self._rag(chosen, user_text)[1]
            return {"type":"answer","chosen_agent":chosen,"meta":meta,"text":self._answer(chosen,user_text,h),"used_chunks":h}

        # Relaxed routing before giving up
        if ROUTE_FALLBACK != "off":
            rc, rmeta = self.router.choose_agent_relaxed(user_text)
            if rc != 0:
                h = rmeta.get("hits") or []
                return {"type":"answer","chosen_agent":rc,"meta":rmeta,"text":self._answer(rc,user_text,h),"used_chunks":h}

        # Honest fallback
        return {
            "type":"answer","chosen_agent":0,
            "meta":{"route":"no-strong-evidence","rag_top_president": top0, "threshold": RAG_MIN_SCORE},
            "text": self._answer(0, user_text, []) + "\n\n(Confidence is low. Add or upload relevant docs for better context.)",
            "used_chunks":[]
        }
