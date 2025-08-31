# app/celery_app.py
import os, platform
from celery import Celery
from app.config import REDIS_URL

def make_celery():
    app = Celery(
        "president",
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=["workers.tasks"],
    )
    app.conf.task_routes = {"workers.*": {"queue": "rag"}}

    # Windows: prefer solo pool unless you override via env
    if platform.system() == "Windows" and not os.getenv("CELERY_POOL"):
        app.conf.worker_pool = "solo"
        app.conf.worker_concurrency = 1
        # Optional: can reduce noise
        # app.conf.worker_hijack_root_logger = False
    return app

celery_app = make_celery()
celery = celery_app
#docker compose up -d
#celery -A app.celery_app.celery worker --loglevel=info --pool=solo -Q rag
#ollama run llama3
#$env:PYTHONPATH = (Get-Location).Path
#python -m uvicorn app.api:app --reload