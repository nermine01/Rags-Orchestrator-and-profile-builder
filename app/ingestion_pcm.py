# app/ingestion_pcm.py
import os
import json
import numpy as np

try:
    import faiss  # pip install faiss-cpu  (on Windows)
except ImportError as e:
    raise ImportError(
        "FAISS not installed. On Windows run: pip install faiss-cpu"
    ) from e

from sqlalchemy import text as sqltext
from app.config import RAG_INDEX_ROOT
from orchestrator.llm import Embeddings
from app.db import SessionLocal


def _index_dir(agent_id: int) -> str:
    return os.path.join(RAG_INDEX_ROOT, f"agent{agent_id}_index")


def _index_path(index_dir: str) -> str:
    return os.path.join(index_dir, "index.faiss")


def _load_or_create_index(dim: int, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    p = _index_path(index_dir)
    if os.path.exists(p):
        return faiss.read_index(p)
    # cosine similarity (normalize vectors first)
    return faiss.IndexFlatIP(dim)


def _save_index(index, index_dir: str):
    faiss.write_index(index, _index_path(index_dir))


def _append_meta(index_dir: str, metas: list[dict]):
    meta_path = os.path.join(index_dir, "meta.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def _store_centroid(agent_id: int, vectors: np.ndarray):
    """Compute centroid from newly-added vectors and write to DB (bytea)."""
    if vectors.size == 0:
        return
    faiss.normalize_L2(vectors)
    centroid = vectors.mean(axis=0, keepdims=True).astype("float32")
    faiss.normalize_L2(centroid)

    with SessionLocal() as s:
        try:
            # Ensure agents has a 'centroid' BYTEA column
            s.execute(
                sqltext("UPDATE agents SET centroid = :c WHERE id = :i"),
                {"c": centroid.tobytes(), "i": agent_id},
            )
            s.commit()
        except Exception as e:
            s.rollback()
            # Non-fatal: index is still updated; routing may be weaker without centroid
            print(f"[ingestion_pcm] Warning: couldn't write centroid for agent {agent_id}: {e}")


def ingest_short_texts_for_agent(agent_id: int, docs: list[tuple[str, str]]):
    """
    docs: list of (source_name, text)
    - embeds texts with your project's Embeddings()
    - appends to agent's FAISS index
    - writes simple meta
    - recomputes/stores centroid from the new vectors
    """
    texts = [t for _, t in docs if (t or "").strip()]
    if not texts:
        return

    index_dir = _index_dir(agent_id)
    embedder = Embeddings()  # uses your project's embedding backend

    # shape: (n_docs, dim)
    vecs = np.array(embedder.embed(texts), dtype="float32")
    faiss.normalize_L2(vecs)

    index = _load_or_create_index(vecs.shape[1], index_dir)
    index.add(vecs)
    _save_index(index, index_dir)

    metas = [{"source": src, "len": len(t or ""), "kind": "pcm"} for src, t in docs]
    _append_meta(index_dir, metas)

    _store_centroid(agent_id, vecs)
