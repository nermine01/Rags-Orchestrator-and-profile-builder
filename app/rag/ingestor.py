import os, tempfile, json
from datetime import datetime
from pdfminer.high_level import extract_text as pdf_extract_text
from app.config import RAG_INDEX_ROOT, EMBED_MODEL
from app.models import Document

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    faiss = None; SentenceTransformer = None; np = None

class Ingestor:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL) if SentenceTransformer else None

    @staticmethod
    def index_dir_for_agent(agent_id: int) -> str:
        return os.path.join(RAG_INDEX_ROOT, f"agent{agent_id}_index")

    def ensure_index_dir(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)

    def _load_pdf_bytes(self, doc: Document) -> bytes:
        if doc.file_data: return bytes(doc.file_data)
        if doc.file_path and os.path.exists(doc.file_path):
            with open(doc.file_path, "rb") as f: return f.read()
        raise RuntimeError(f"No PDF content for doc {doc.id}")

    def _extract_text(self, b: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.write(b); tmp.flush()
            return pdf_extract_text(tmp.name) or ""
        finally:
            name = tmp.name
            try: tmp.close()
            finally:
                try: os.unlink(name)
                except: pass

    def _chunk(self, text: str, size=800, overlap=200):
        text = text.strip()
        if not text: return []
        out, i, n = [], 0, len(text)
        while i < n:
            j = min(i+size, n)
            out.append(text[i:j])
            i = j - overlap if j - overlap > i else j
        return out

    def _faiss_paths(self, index_dir: str):
        return os.path.join(index_dir, "index.faiss"), os.path.join(index_dir, "meta.jsonl")

    def _faiss_upsert(self, index_dir, vectors, metas):
        index_path, meta_path = self._faiss_paths(index_dir)
        d = vectors.shape[1]
        if os.path.exists(index_path): index = faiss.read_index(index_path)
        else: index = faiss.IndexFlatIP(d)
        norms = (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        vectors = vectors / norms
        index.add(vectors)
        faiss.write_index(index, index_path)
        with open(meta_path, "a", encoding="utf-8") as f:
            for m in metas: f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def ingest(self, db, doc: Document, index_dir: str):
        pdf = self._load_pdf_bytes(doc)
        text = self._extract_text(pdf)
        open(os.path.join(index_dir, f"doc_{doc.id}.txt"), "w", encoding="utf-8").write(text)

        chunks = self._chunk(text)
        if chunks and self.model and faiss and np is not None:
            vecs = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            metas = [{"doc_id": doc.id, "chunk": i} for i in range(len(chunks))]
            self._faiss_upsert(index_dir, vecs, metas)

            # --- update centroid (simple mean of this doc) ---
            from app.db import SessionLocal
            from app.models import Agent
            with SessionLocal() as _db:
                ag = _db.get(Agent, doc.agent_id)
                if ag:
                    centroid = vecs.mean(axis=0).astype("float32")
                    ag.centroid = centroid.tobytes()
                    _db.add(ag); _db.commit()

        doc.status = True
        doc.index_path = index_dir
        doc.indexed_at = datetime.utcnow()
        doc.last_error = None
        db.add(doc)

    def finalize(self, index_dir: str): pass
