# 🏛️ President Orchestrator

This project is a multi-agent question-answering system with an orchestrator that decides which agent should answer a user query. Fully local, using FAISS for retrieval and Ollama for generation. Includes an optional PCM (Process Communication Model) pipeline for video-based personality/emotion profiling that ingests its results as a dedicated agent’s memory.

---

## 🚀 What This Project Is

The system consists of:

- **President agent (ID 0)**: answers first using its own memory via RAG.  
- **Specialist agents (IDs > 0)**: each has a dedicated RAG memory.  
- **Orchestrator / Router**: decides whether President handles a query or delegates to another agent.  
- **Local LLM** (via Ollama): generates grounded answers from retrieved context.  
- **HTTP API** (FastAPI): chat and admin/debug endpoints.  
- **Pluggable ingestion pipeline**: turns PDFs into per-agent FAISS indices.

Everything runs locally: embeddings, search, and generation.

---

## 🔄 Request Flow

1. Client calls `/api/chat` with a question (optionally `agent_id`).  
2. **President attempts first**:
   - Vectorizes query and searches its FAISS index.
   - If top match ≥ `RAG_MIN_SCORE`, it answers directly.
3. **Router evaluates other agents** (if President is weak):
   - Shortlists candidates using **centroids**.
   - Probes shortlisted agents’ FAISS indices.
   - Chooses agent with top RAG score ≥ strict threshold.
   - If none meet strict threshold, optional **relaxed fallback** selects best available agent.
4. Selected agent generates answer via Ollama, grounded in retrieved chunks.  
5. API returns JSON including:
   - Chosen agent
   - Routing path
   - Thresholds
   - Evidence (top chunks and scores)

<details>
<summary>Click to expand: Routing Logic & Fallback</summary>

- **Strict pass**: Only agents with RAG ≥ `RAG_MIN_SCORE` are eligible.  
- **Relaxed fallback**: If no agent passes strict, options:
  - `best_rag`: pick agent with highest RAG above `ROUTE_FALLBACK_MIN_RAG`.  
  - `centroid`: pick agent with highest centroid similarity above `ROUTE_FALLBACK_MIN_CENTROID`.  
  - `off`: no relaxed routing; stays with President.
</details>

---

## 🧩 Core Components

### 1️⃣ President (Agent 0)
- Holds FAISS index (“memory”).  
- Answers directly if top retrieval score is strong.  
- Delegates to router otherwise.

### 2️⃣ Specialist Agents (IDs > 0)
- Each agent has its own FAISS index in `agent{ID}_index`.  
- Router uses centroids to shortlist likely candidates.

### 3️⃣ Router / Orchestrator
- Embeds query same as RAG.  
- Performs strict pass and optional relaxed fallback.  

### 4️⃣ RAG Stack
- **Embeddings**: sentence-transformers model (configurable).  
- **Indexing**: FAISS per agent; metadata tracks chunk origins.  
- **Retrieval**: cosine/inner-product similarity; returns top chunks.

### 5️⃣ LLM (Generation)
- **Ollama** used for completion.  
- Supports `/api/chat` and `/api/generate`.  
- Configurable model tag (local Llama 3 variant, for example).  
- Optional fallback to OpenAI if Ollama unavailable.  

### 6️⃣ HTTP API
- **Chat endpoint**: returns chosen agent, route, thresholds, evidence.  
- **Admin endpoints**:
  - Health: index existence, chunk counts, centroid presence.  
  - Router explain: centroid similarities and per-agent RAG scores.  
- **PCM endpoints**: video analysis → traits, emotions, transcript, guidelines → ingested into PCM agent.

---

## 🎭 PCM Profiling (Separate Conda Env)

- Runs in `newenv` to isolate heavy audio/vision deps (pyannote, Whisper, OpenCV).  
- Extracts audio → transcribes → diarizes → estimates audio & visual emotions → scores communication styles → builds PCM profiles → summarizes → generates communication guidelines via LLM.  
- Produces compact “short texts” embedded into PCM agent’s FAISS index.  
- API calls PCM worker via: run -n newenv python -m pcm_worker.cli ...

---

## 💾 Data & Storage

### Database (Logical)
- **Agents table**: ID, name, description, centroid (float32 vector).  
- **Documents table** (optional): metadata, agent association, status (processed/unprocessed).

### Filesystem Layout
- Root storage directory contains per-agent FAISS indices: `agent{ID}_index`.  
- Each index directory includes:
  - FAISS index file
  - Metadata file (document ID, page, offsets…)

### Ingestion Patterns
1. **Manual batch ingestion**:
   - Drop PDFs for an agent → run ingestor → updates FAISS index & centroid.
2. **DB-driven ingestion**:
   - Add new row (PDF) with `processed = false`.  
   - Background ingestor updates index & centroid automatically.

> ✅ Chunking strategy and embedding model are consistent across the system.

---

## ⚙️ Configuration (Environment Variables)

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | Usually `"ollama"` |
| `OLLAMA_BASE_URL` | URL for local Ollama server |
| `OLLAMA_MODEL` | Local model tag for answers |
| `EMBED_MODEL` | Sentence-transformers model for embeddings |
| `RAG_INDEX_ROOT` | Absolute path to FAISS index root |
| `RAG_MIN_SCORE` | Strict threshold for confident retrieval |
| `ROUTE_FALLBACK` | Relaxed fallback policy (`best_rag`, `centroid`, `off`) |
| `ROUTE_FALLBACK_MIN_RAG` | Relaxed RAG floor |
| `ROUTE_FALLBACK_MIN_CENTROID` | Relaxed centroid floor |
| `DATABASE_URL` | DSN for Agents table and centroids |

---

## 📝 Operational Notes

- **Health & debugging**: `/api/admin/agents_health`, `/api/admin/router_explain`.  
- **Windows specifics**: use absolute paths for `RAG_INDEX_ROOT`.  
- **Performance tuning**: thresholds, fallback policies, embedding models, index hygiene.  
- **Security & privacy**: fully local; protect storage directory; raw document text not returned.

<details>
<summary>Click to expand: Adding Agents & Tools</summary>

- **Add a new agent**: create DB entry, ingest documents, compute centroid.  
- **Add tools/skills**: agent can call tools; router is agnostic.  
- **Change routing logic**: strict and relaxed phases are isolated; you can add heuristics.
</details>

---

## ✅ Done Looks Like

- President answers confidently when it has memory.  
- Router delegates correctly to specialists.  
- Agents have FAISS index & centroid visible in health endpoint.  
- LLM answers are grounded, concise, and include routing metadata.  
- Ingestion pipelines keep memories up to date.

---

## 📚 Glossary

- **Agent**: answering unit with memory (FAISS index).  
- **RAG**: Retrieval-Augmented Generation.  
- **FAISS**: Local vector index for similarity search.  
- **Centroid**: vector summarizing an agent’s memory.  
- **Strict routing**: only agents meeting `RAG_MIN_SCORE`.  
- **Relaxed routing**: optional fallback when strict fails.
