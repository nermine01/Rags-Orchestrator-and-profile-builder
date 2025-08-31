from sqlalchemy import event
from app.models import Agent, Document
from app.config import SYNC_INGEST

TASK_NAME = "workers.tasks.ingest_pending_docs_for_agent"

def _dispatch(agent_id: int):
    if SYNC_INGEST:
        from workers.tasks import ingest_pending_docs_for_agent
        ingest_pending_docs_for_agent.apply(args=[agent_id])
    else:
        from app.celery_app import celery_app
        celery_app.send_task(TASK_NAME, args=[agent_id])

@event.listens_for(Agent, "after_insert")
def _agent_after_insert(mapper, connection, target):
    _dispatch(target.id)

@event.listens_for(Document, "after_insert")
def _doc_after_insert(mapper, connection, target):
    if target.agent_id and (target.status is False):
        _dispatch(target.agent_id)

@event.listens_for(Document, "after_update")
def _doc_after_update(mapper, connection, target):
    if target.agent_id and (target.status is False):
        _dispatch(target.agent_id)
