import os, json
from typing import List, Tuple
from app.rag.ingestor import Ingestor
try:
    import faiss, numpy as np
except Exception:
    faiss = None; np = None

class Retriever:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.index_dir = Ingestor.index_dir_for_agent(agent_id)
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.meta_path  = os.path.join(self.index_dir, "meta.jsonl")
        self.index = faiss.read_index(self.index_path) if (faiss and os.path.exists(self.index_path)) else None
        self.metas = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metas = [json.loads(line) for line in f]

    def search(self, query_vec, top_k=4) -> List[Tuple[float, dict]]:
        if not self.index or np is None: return []
        q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True)+1e-12)
        D, I = self.index.search(q, top_k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if 0 <= idx < len(self.metas):
                out.append((float(score), self.metas[idx]))
        return out
