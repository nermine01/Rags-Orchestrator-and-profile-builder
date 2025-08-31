from __future__ import annotations

import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ---------- Helpers ----------
def _project_root() -> str:
    """Return project root (two levels up from this file)."""
    # .../president/app/config.py -> .../president
    return str(pathlib.Path(__file__).resolve().parents[1])


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


def _env_choice(key: str, choices: set[str], default: str) -> str:
    val = (os.getenv(key) or "").strip().lower()
    return val if val in choices else default


# ---------- Core LLM / Embeddings ----------
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3:latest")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")


# ---------- RAG / Index paths ----------
RAG_INDEX_ROOT   = os.getenv("RAG_INDEX_ROOT", os.path.join(_project_root(), "storage"))
RAG_MIN_SCORE    = _env_float("RAG_MIN_SCORE", 0.6)


# ---------- Orchestration fallback knobs ----------
# If nobody reaches RAG_MIN_SCORE, pick someone anyway (relaxed routing)
# choices: "best_rag" | "centroid" | "off"
ROUTE_FALLBACK              = _env_choice("ROUTE_FALLBACK", {"best_rag", "centroid", "off"}, "best_rag")
ROUTE_FALLBACK_MIN_RAG      = _env_float("ROUTE_FALLBACK_MIN_RAG", 0.20)
ROUTE_FALLBACK_MIN_CENTROID = _env_float("ROUTE_FALLBACK_MIN_CENTROID", 0.25)


# ---------- Optional: DB / Broker ----------
DATABASE_URL    = os.getenv("DATABASE_URL") or "sqlite:///./app.db"
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ---------- Integrations ----------
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")
SYNC_INGEST     = os.getenv("SYNC_INGEST", "0") == "1"

#------------ pcm agent -----------
PCM_AGENT_NAME = os.getenv("PCM_AGENT_NAME", "PCM Agent")
PCM_FASTPATH_KEYWORDS = [s.strip().lower() for s in os.getenv(
    "PCM_FASTPATH_KEYWORDS",
    "pcm,profile my call,communication style,diarization,emotion,nonverbal,tone"
).split(",")]

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # or "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
AGENTS_TABLE = os.getenv("AGENTS_TABLE", "agents")