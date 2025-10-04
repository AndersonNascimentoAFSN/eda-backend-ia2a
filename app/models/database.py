"""
Modelos de banco de dados para análises EDA
"""
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class para todos os modelos"""
    pass


class Analysis(Base):
    """Modelo para análises EDA"""
    __tablename__ = "analyses"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    file_key: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    analysis_type: Mapped[str] = mapped_column(String(100), nullable=False, default="basic_eda")
    progress: Mapped[float] = mapped_column(nullable=False, default=0.0)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Dados da análise
    results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    visualizations: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Metadados do arquivo
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_format: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    rows_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Configurações
    options: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Controle
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    def __repr__(self) -> str:
        return f"Analysis(id={self.id!r}, file_key={self.file_key!r}, status={self.status!r})"


class AnalysisSession(Base):
    """Modelo para sessões de análise (WebSocket)"""
    __tablename__ = "analysis_sessions"
    
    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), nullable=False)
    client_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Status da sessão
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_ping: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    def __repr__(self) -> str:
        return f"AnalysisSession(session_id={self.session_id!r}, analysis_id={self.analysis_id!r})"