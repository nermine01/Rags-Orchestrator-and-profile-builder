# orchestrator/llm.py
from __future__ import annotations
import threading
from typing import List
import httpx
import numpy as np

# App config
from app.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    EMBED_MODEL,          # make sure this exists in app.config (see step 2)
)

# Lazy import for heavy deps
_sentence_model = None
_sentence_model_name = None
_sentence_model_lock = threading.Lock()


class Embeddings:
    """
    Thin wrapper around Sentence-Transformers with row-wise L2 normalization.
    Returns float32 numpy arrays suitable for cosine / inner-product search.
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBED_MODEL or "sentence-transformers/all-MiniLM-L6-v2"

    def _ensure_model(self):
        global _sentence_model, _sentence_model_name
        if _sentence_model is not None and _sentence_model_name == self.model_name:
            return
        with _sentence_model_lock:
            if _sentence_model is not None and _sentence_model_name == self.model_name:
                return
            from sentence_transformers import SentenceTransformer  # import here to avoid import cost on API boot
            _sentence_model = SentenceTransformer(self.model_name)
            _sentence_model_name = self.model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        vecs = _sentence_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # we normalize manually just below
        ).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return (vecs / norms).astype("float32")
    def embed(self, texts):
        return self.encode(texts)
    def embed_one(self, text):
        return self.encode_one(text)

class LLM:
    """
    LLM completion client. Uses Ollama when LLM_PROVIDER=ollama.
    Falls back from /api/chat to /api/generate for older Ollama servers.
    """
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.api_key = OPENAI_API_KEY
        self.ollama_base = OLLAMA_BASE_URL
        self.ollama_model = OLLAMA_MODEL

    # -------- Ollama helpers --------
    def _ollama_chat(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.ollama_base}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": "Be concise. Ground answers in the provided context."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        if r.status_code == 404:
            raise FileNotFoundError("ollama_chat_not_available")
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "")

    def _ollama_generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.ollama_base}/api/generate",
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        # /api/generate returns {"response": "..."} when stream=False
        return data.get("response") or (data.get("message") or {}).get("content", "")

    # -------- Public API --------
    def complete(self, prompt: str) -> str:
        if self.provider == "ollama":
            try:
                return self._ollama_chat(prompt)
            except FileNotFoundError:
                return self._ollama_generate(prompt)
            except Exception as e:
                return f"[OLLAMA ERROR] {e}\n{prompt[:400]}"
        # Add other providers if you keep them; default fallback:
        return f"[PRESIDENT DUMMY]\n{prompt[:400]}"
