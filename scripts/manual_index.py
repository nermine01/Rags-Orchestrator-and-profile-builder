# scripts/manual_index.py
import os, sys, argparse, glob, json, tempfile
from typing import List, Tuple

# --- Optional: allow importing your app.* modules if you want DB centroid update
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# PDF → text
from pdfminer.high_level import extract_text as pdf_extract_text

# Embeddings / FAISS
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Optional DB centroid update
def update_db_centroid(agent_id: int, centroid_vec: np.ndarray) -> None:
    try:
        from app.db import SessionLocal
        from app.models import Agent
    except Exception as e:
        print(f"[warn] DB imports unavailable; skipping centroid update: {e}")
        return
    with SessionLocal() as db:
        ag = db.get(Agent, agent_id)
        if ag is None:
            print(f"[warn] Agent {agent_id} not found; skipping centroid update")
            return
        ag.centroid = centroid_vec.astype("float32").tobytes()
        db.add(ag); db.commit()
        print(f"[ok] Updated DB centroid for agent {agent_id} ({centroid_vec.shape[0]} dims)")

def chunk_text(text: str, size=800, overlap=200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        out.append(text[i:j])
        i = j - overlap if j - overlap > i else j
    return out

def extract_pdf_text(path: str) -> str:
    return pdf_extract_text(path) or ""

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def faiss_paths(index_dir: str) -> Tuple[str, str]:
    return os.path.join(index_dir, "index.faiss"), os.path.join(index_dir, "meta.jsonl")

def open_index(index_path: str, dim: int):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    return index

def append_vectors(index, index_path: str, vectors: np.ndarray):
    # normalize row-wise for cosine similarity
    norms = (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    vectors = vectors / norms
    index.add(vectors)
    faiss.write_index(index, index_path)

def next_doc_id(meta_path: str) -> int:
    max_id = -1
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    did = int(obj.get("doc_id", -1))
                    if did > max_id: max_id = did
                except Exception:
                    pass
    return max_id + 1

def save_metas(meta_path: str, metas: List[dict]):
    with open(meta_path, "a", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def write_doc_text(index_dir: str, doc_id: int, text: str):
    with open(os.path.join(index_dir, f"doc_{doc_id}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

def list_pdfs(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        # common PDF patterns
        return sorted(glob.glob(os.path.join(input_path, "*.pdf")))
    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        return [input_path]
    raise FileNotFoundError(f"No PDFs found at: {input_path}")

def main():
    ap = argparse.ArgumentParser(description="Manual RAG index builder (append/rebuild) for an agent.")
    ap.add_argument("--agent", type=int, required=True, help="Agent ID (use 0 for President memory).")
    ap.add_argument("--input", type=str, required=True, help="Folder of PDFs or a single PDF file.")
    ap.add_argument("--index-root", type=str, default=os.getenv("RAG_INDEX_ROOT", "./storage"),
                    help="Root directory for indexes (default from RAG_INDEX_ROOT or ./storage).")
    ap.add_argument("--mode", choices=["append", "rebuild"], default="append",
                    help="Append to existing index or rebuild from scratch.")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--embed-model", type=str, default=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--update-db-centroid", action="store_true",
                    help="After adding, compute centroid for JUST added chunks and write to DB (optional).")
    ap.add_argument("--recompute-centroid", action="store_true",
                    help="Compute centroid by re-embedding all doc_*.txt in the index folder and write to DB.")
    args = ap.parse_args()

    agent_id = args.agent
    index_dir = os.path.join(args.index_root, f"agent{agent_id}_index")
    ensure_dir(index_dir)
    index_path, meta_path = faiss_paths(index_dir)

    # (re)build?
    if args.mode == "rebuild":
        for p in [index_path, meta_path]:
            if os.path.exists(p):
                os.remove(p)
        # also remove doc_*.txt to keep it clean
        for fn in glob.glob(os.path.join(index_dir, "doc_*.txt")):
            try: os.remove(fn)
            except: pass
        print(f"[rebuild] Cleared previous index at {index_dir}")

    pdfs = list_pdfs(args.input)
    if not pdfs:
        print("[done] no PDFs to index"); return

    print(f"[info] Agent {agent_id} | PDFs: {len(pdfs)}")
    print(f"[info] Index dir: {os.path.abspath(index_dir)}")
    print(f"[info] Model: {args.embed_model}")

    model = SentenceTransformer(args.embed_model)

    # open/create index lazily (after we know dims)
    index = None
    all_added_vecs = []  # for optional centroid update of just-added content

    for pdf_path in pdfs:
        print(f"[pdf] {pdf_path}")
        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text, size=args.chunk_size, overlap=args.chunk_overlap)
        if not chunks:
            print("  [skip] empty text")
            continue

        # embed
        vecs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        dim = int(vecs.shape[1])

        if index is None:
            index = open_index(index_path, dim)

        # append vectors
        append_vectors(index, index_path, vecs)

        # update meta.jsonl
        did = next_doc_id(meta_path)
        metas = [{"doc_id": did, "chunk": i, "source": os.path.basename(pdf_path)} for i in range(len(chunks))]
        save_metas(meta_path, metas)

        # save raw text (debug/useful later)
        write_doc_text(index_dir, did, text)

        all_added_vecs.append(vecs)
        print(f"  [ok] doc_id={did} chunks={len(chunks)}")

    if index is None:
        print("[warn] nothing indexed (no non-empty PDFs)")
        return

    # Optional centroid → DB
    if args.update_db_centroid or args.recompute_centroid:
        if args.recompute_centroid:
            # recompute centroid using all doc_*.txt in this index_dir
            texts = []
            doc_txts = sorted(glob.glob(os.path.join(index_dir, "doc_*.txt")))
            for fp in doc_txts:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        t = f.read()
                    texts.extend(chunk_text(t, size=args.chunk_size, overlap=args.chunk_overlap))
                except Exception:
                    pass
            if texts:
                vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                centroid = vecs.mean(axis=0)
                update_db_centroid(agent_id, centroid)
            else:
                print("[warn] recompute-centroid: no text found to compute centroid")
        else:
            # centroid of just-added chunks (approximation)
            if all_added_vecs:
                vecs = np.vstack(all_added_vecs)
                centroid = vecs.mean(axis=0)
                update_db_centroid(agent_id, centroid)
            else:
                print("[warn] update-db-centroid: no new vectors added")

    print("[done] indexing complete")
    # quick summary
    files = os.listdir(index_dir)
    print(" ->", os.path.abspath(index_dir))
    for fn in files:
        print("    ", fn)

if __name__ == "__main__":
    main()
