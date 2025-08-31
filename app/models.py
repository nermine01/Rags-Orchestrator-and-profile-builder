from datetime import datetime
from sqlalchemy.orm import declarative_base, relationship, mapped_column, Mapped
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, LargeBinary
from sqlalchemy.types import JSON, Text, Integer, DateTime
Base = declarative_base()

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    centroid = Column(LargeBinary)  # router centroid
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    documents = relationship("Document", back_populates="agent")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), index=True, nullable=False)
    mime = Column(String(100), default="application/pdf")
    status = Column(Boolean, default=False, nullable=False)
    file_path = Column(String(1024))
    file_data = Column(LargeBinary)
    index_path = Column(String(1024))
    indexed_at = Column(DateTime)
    last_error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    agent = relationship("Agent", back_populates="documents")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), index=True, nullable=False)
    user_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tool_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
from sqlalchemy.dialects.postgresql import JSONB

class PCMAnalysis(Base):
    __tablename__ = "pcm_analysis"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str | None] = mapped_column(Text)
    transcript: Mapped[str | None] = mapped_column(Text)
    vectors: Mapped[dict | None] = mapped_column(JSON)
    scripts: Mapped[dict | None] = mapped_column(JSON)
    emotions: Mapped[dict | list | None] = mapped_column(JSON)
    summary: Mapped[str | None] = mapped_column(Text)
    guideline: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)