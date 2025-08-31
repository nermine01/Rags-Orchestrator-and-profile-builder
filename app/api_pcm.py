import os, json, subprocess, shutil, tempfile
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from tempfile import NamedTemporaryFile
from app.db import SessionLocal
from app.models import PCMAnalysis
from app.ingestion_pcm import ingest_short_texts_for_agent
from app.config import PCM_AGENT_NAME
from sqlalchemy import text as sqltext

router = APIRouter(prefix="/pcm", tags=["pcm"])


def _conda_exe() -> str:
    # Adjust to your install if needed
    cand = shutil.which("conda") or r"C:\ProgramData\miniconda3\condabin\conda.bat"
    return cand

def _run_pcm_in_newenv(video_path: str) -> dict:
    conda = _conda_exe()
    if not conda:
        raise RuntimeError("conda not found on PATH")

    # temp file to receive JSON result
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    out_path = out_file.name
    out_file.close()

    env = os.environ.copy()
    # If pyannote needs token, make sure it's set in env before launching FastAPI,
    # e.g. $env:HUGGINGFACE_TOKEN="hf_..." in your PowerShell
    # Same for OPENAI_API_KEY if used inside newenv.

    try:
        # On Windows, conda is often a .bat => call via cmd.exe /c
        if conda.lower().endswith(".bat"):
            cmd = f'"{conda}" run -n newenv python -m pcm_worker.cli --input "{video_path}" --out "{out_path}"'
            res = subprocess.run(cmd, capture_output=True, text=True, shell=True, env=env)
        else:
            cmd = [
                conda, "run", "-n", "newenv",
                "python", "-m", "pcm_worker.cli",
                "--input", video_path,
                "--out", out_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if res.returncode != 0:
            raise RuntimeError(
                f"[PCM] Subprocess failed (code {res.returncode}). "
                f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
            )

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            # nothing written; include stdout/stderr for diagnostics
            raise RuntimeError(
                "[PCM] Outfile missing or empty.\n"
                f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
            )

        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass

def _get_pcm_agent_id() -> int:
    with SessionLocal() as s:
        # look up by name
        res = s.execute(
            sqltext("SELECT id FROM agents WHERE name = :n LIMIT 1"),
            {"n": PCM_AGENT_NAME},
        )
        row = res.fetchone()
        if row:
            return row[0]

        # create if missing
        res = s.execute(
            sqltext("INSERT INTO agents (name, description) VALUES (:n, :d) RETURNING id"),
            {"n": PCM_AGENT_NAME, "d": "PCM profiling specialist"},
        )
        new_id = res.scalar_one()
        s.commit()
        return new_id


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), background: BackgroundTasks = None):
    agent_id = _get_pcm_agent_id()

    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def _job():
        try:
            data = _run_pcm_in_newenv(tmp_path)
        finally:
            try: os.remove(tmp_path)
            except: pass

        # persist
        with SessionLocal() as s:
            row = PCMAnalysis(
                agent_id=agent_id,
                filename=file.filename,
                transcript=data.get("transcript"),
                vectors=data.get("profiles"),
                scripts=data.get("scripts"),
                emotions=data.get("emotions"),
                summary=data.get("summary"),
                guideline=data.get("guideline"),
            )
            s.add(row); s.commit(); s.refresh(row)
            analysis_id = row.id

        # feed RAG with concise docs
        docs = [
            (f"pcm_summary_{analysis_id}.txt", data.get("summary","")),
            (f"pcm_guideline_{analysis_id}.txt", data.get("guideline","")),
        ]
        ingest_short_texts_for_agent(agent_id, docs)

    background.add_task(_job)
    return {"status": "started"}
from fastapi import Path

@router.get("/analysis/{analysis_id}")
def get_pcm_analysis(analysis_id: int = Path(..., description="ID of the PCM analysis")):
    with SessionLocal() as s:
        row = s.query(PCMAnalysis).filter(PCMAnalysis.id == analysis_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return {
            "id": row.id,
            "agent_id": row.agent_id,
            "filename": row.filename,
            "transcript": row.transcript,
            "summary": row.summary,
            "guideline": row.guideline,
            "emotions": row.emotions,
            "scripts": row.scripts,
            "vectors": row.vectors,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
