"""
Repositório para operações de banco de dados das análises
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.database import Analysis, AnalysisSession


class AnalysisRepository:
    """Repositório para operações de análises"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_analysis(
        self,
        analysis_id: str,
        file_key: str,
        analysis_type: str = "basic_eda",
        options: Optional[Dict[str, Any]] = None
    ) -> Analysis:
        """Criar nova análise"""
        analysis = Analysis(
            id=analysis_id,
            file_key=file_key,
            analysis_type=analysis_type,
            status="pending",
            progress=0.0,
            options=options or {},
            created_at=datetime.utcnow()
        )
        
        self.session.add(analysis)
        await self.session.commit()
        await self.session.refresh(analysis)
        return analysis
    
    async def get_analysis(self, analysis_id: str) -> Optional[Analysis]:
        """Obter análise por ID"""
        stmt = select(Analysis).where(
            and_(
                Analysis.id == analysis_id,
                Analysis.is_deleted == False
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
    
    async def update_analysis_status(
        self,
        analysis_id: str,
        status: str,
        progress: float = None,
        message: str = None,
        started_at: datetime = None,
        completed_at: datetime = None
    ) -> bool:
        """Atualizar status da análise"""
        update_data = {"status": status}
        
        if progress is not None:
            update_data["progress"] = progress
        if message is not None:
            update_data["message"] = message
        if started_at is not None:
            update_data["started_at"] = started_at
        if completed_at is not None:
            update_data["completed_at"] = completed_at
        
        stmt = update(Analysis).where(Analysis.id == analysis_id).values(**update_data)
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount > 0
    
    async def update_analysis_results(
        self,
        analysis_id: str,
        results: Dict[str, Any] = None,
        summary: Dict[str, Any] = None,
        visualizations: Dict[str, Any] = None,
        file_metadata: Dict[str, Any] = None
    ) -> bool:
        """Atualizar resultados da análise"""
        update_data = {}
        
        if results is not None:
            update_data["results"] = results
        if summary is not None:
            update_data["summary"] = summary
        if visualizations is not None:
            update_data["visualizations"] = visualizations
        
        if file_metadata:
            if "file_size" in file_metadata:
                update_data["file_size"] = file_metadata["file_size"]
            if "file_format" in file_metadata:
                update_data["file_format"] = file_metadata["file_format"]
            if "rows_count" in file_metadata:
                update_data["rows_count"] = file_metadata["rows_count"]
            if "columns_count" in file_metadata:
                update_data["columns_count"] = file_metadata["columns_count"]
        
        stmt = update(Analysis).where(Analysis.id == analysis_id).values(**update_data)
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount > 0
    
    async def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str = None,
        file_key_pattern: str = None
    ) -> List[Analysis]:
        """Listar análises com filtros"""
        stmt = select(Analysis).where(Analysis.is_deleted == False)
        
        if status:
            stmt = stmt.where(Analysis.status == status)
        
        if file_key_pattern:
            stmt = stmt.where(Analysis.file_key.like(f"%{file_key_pattern}%"))
        
        stmt = stmt.order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def delete_analysis(self, analysis_id: str, soft_delete: bool = True) -> bool:
        """Deletar análise (soft ou hard delete)"""
        if soft_delete:
            stmt = update(Analysis).where(Analysis.id == analysis_id).values(
                is_deleted=True,
                completed_at=datetime.utcnow()
            )
        else:
            stmt = delete(Analysis).where(Analysis.id == analysis_id)
        
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount > 0
    
    async def cleanup_old_analyses(self, days_old: int = 7) -> int:
        """Limpar análises antigas"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Soft delete de análises antigas
        stmt = update(Analysis).where(
            and_(
                Analysis.created_at < cutoff_date,
                Analysis.is_deleted == False
            )
        ).values(is_deleted=True)
        
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount


class AnalysisSessionRepository:
    """Repositório para sessões WebSocket"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_session(
        self,
        session_id: str,
        analysis_id: str,
        client_id: str = None
    ) -> AnalysisSession:
        """Criar nova sessão"""
        session_obj = AnalysisSession(
            session_id=session_id,
            analysis_id=analysis_id,
            client_id=client_id,
            is_active=True,
            last_ping=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        self.session.add(session_obj)
        await self.session.commit()
        await self.session.refresh(session_obj)
        return session_obj
    
    async def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Obter sessão por ID"""
        stmt = select(AnalysisSession).where(AnalysisSession.session_id == session_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()
    
    async def update_session_ping(self, session_id: str) -> bool:
        """Atualizar último ping da sessão"""
        stmt = update(AnalysisSession).where(
            AnalysisSession.session_id == session_id
        ).values(last_ping=datetime.utcnow())
        
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount > 0
    
    async def get_active_sessions_for_analysis(self, analysis_id: str) -> List[AnalysisSession]:
        """Obter sessões ativas para uma análise"""
        stmt = select(AnalysisSession).where(
            and_(
                AnalysisSession.analysis_id == analysis_id,
                AnalysisSession.is_active == True
            )
        )
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def close_session(self, session_id: str) -> bool:
        """Fechar sessão"""
        stmt = update(AnalysisSession).where(
            AnalysisSession.session_id == session_id
        ).values(
            is_active=False,
            ended_at=datetime.utcnow()
        )
        
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount > 0
    
    async def cleanup_inactive_sessions(self, hours_old: int = 24) -> int:
        """Limpar sessões inativas"""
        cutoff_date = datetime.utcnow() - timedelta(hours=hours_old)
        
        stmt = update(AnalysisSession).where(
            and_(
                AnalysisSession.last_ping < cutoff_date,
                AnalysisSession.is_active == True
            )
        ).values(
            is_active=False,
            ended_at=datetime.utcnow()
        )
        
        result = await self.session.execute(stmt)
        await self.session.commit()
        
        return result.rowcount