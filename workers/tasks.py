import redis
from contextlib import contextmanager
from celery.utils.log import get_task_logger
from sqlalchemy import select
from app.celery_app import celery_app
from app.db import SessionLocal
from app.models import Agent, Document
from app.rag.ingestor import Ingestor
from app.config import REDIS_URL

logger = get_task_logger(__name__)
rds = redis.from_url(REDIS_URL)

@contextmanager
def redis_lock(key: str, ttl=300, wait=10):
    lock = rds.lock(key, timeout=ttl, blocking_timeout=wait)
    acquired = lock.acquire(blocking=True)
    try: yield acquired
    finally:
        if acquired: lock.release()

@celery_app.task(name="workers.tasks.ingest_pending_docs_for_agent", bind=True, max_retries=5, default_retry_delay=30)
def ingest_pending_docs_for_agent(self, agent_id: int):
    with redis_lock(f"ingest:agent:{agent_id}") as ok:
        if not ok: return
        ing = Ingestor()
        with SessionLocal() as db:
            agent = db.get(Agent, agent_id)
            if not agent: return
            pending = db.execute(select(Document).where(Document.agent_id==agent_id, Document.status==False)).scalars().all()
            if not pending: return
            index_dir = ing.index_dir_for_agent(agent_id); ing.ensure_index_dir(index_dir)
            for doc in pending:
                try:
                    ing.ingest(db, doc, index_dir)
                    db.commit()
                except Exception as e:
                    doc.last_error = str(e)[:2000]; db.add(doc); db.commit()
                    raise self.retry(exc=e)
            ing.finalize(index_dir)
