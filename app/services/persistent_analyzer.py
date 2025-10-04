"""
Serviço de análise de dados com persistência
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import io
import logging

from app.core.database import get_db_session
from app.repositories.analysis_repository import AnalysisRepository
from app.core.websocket import websocket_manager
from app.core.r2_service import r2_service
from app.services.eda_processor import EDAProcessor
# from app.models.data_models import AnalysisStartRequest

logger = logging.getLogger(__name__)


class PersistentDataAnalyzer:
    """Analisador de dados com persistência em banco de dados"""
    
    def __init__(self):
        self.eda_processor = EDAProcessor()
        self._active_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_analysis(
        self,
        file_key: str,
        analysis_type: str = "basic_eda",
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Iniciar análise com persistência"""
        analysis_id = str(uuid.uuid4())
        
        # Salvar no banco de dados
        async for db_session in get_db_session():
            repo = AnalysisRepository(db_session)
            await repo.create_analysis(
                analysis_id=analysis_id,
                file_key=file_key,
                analysis_type=analysis_type,
                options=options
            )
            break
        
        # Executar análise em background
        task = asyncio.create_task(self._run_persistent_analysis(analysis_id))
        self._active_tasks[analysis_id] = task
        
        logger.info(f"Análise persistente iniciada: {analysis_id}")
        return analysis_id
    
    async def _run_persistent_analysis(self, analysis_id: str):
        """Executar análise com atualizações persistentes"""
        try:
            async for db_session in get_db_session():
                repo = AnalysisRepository(db_session)
                
                # Marcar como iniciada
                await repo.update_analysis_status(
                    analysis_id=analysis_id,
                    status="processing",
                    progress=5.0,
                    message="Iniciando análise",
                    started_at=datetime.utcnow()
                )
                
                # Notificar via WebSocket
                await websocket_manager.send_analysis_status(
                    analysis_id, "processing", 5.0, "Iniciando análise"
                )
                
                # Obter dados da análise
                analysis = await repo.get_analysis(analysis_id)
                if not analysis:
                    raise Exception("Análise não encontrada")
                
                # 1. Baixar arquivo do R2
                await repo.update_analysis_status(
                    analysis_id, "processing", 15.0, "Baixando arquivo do R2"
                )
                await websocket_manager.send_analysis_status(
                    analysis_id, "processing", 15.0, "Baixando arquivo do R2"
                )
                
                file_content = await self._download_file_from_r2(analysis.file_key)
                
                # 2. Carregar dados
                await repo.update_analysis_status(
                    analysis_id, "processing", 25.0, "Carregando dados"
                )
                await websocket_manager.send_analysis_status(
                    analysis_id, "processing", 25.0, "Carregando dados"
                )
                
                df = await self._load_dataframe(file_content, analysis.file_key)
                
                # Salvar metadados do arquivo
                file_metadata = {
                    "file_size": len(file_content),
                    "file_format": "csv",
                    "rows_count": len(df),
                    "columns_count": len(df.columns)
                }
                await repo.update_analysis_results(
                    analysis_id, file_metadata=file_metadata
                )
                
                # 3. Executar análise EDA
                await repo.update_analysis_status(
                    analysis_id, "processing", 40.0, "Executando análise EDA"
                )
                await websocket_manager.send_analysis_status(
                    analysis_id, "processing", 40.0, "Executando análise EDA"
                )
                
                eda_results = await self._run_eda_analysis(df, analysis.analysis_type, analysis.options)
                
                # 4. Salvar resultados
                await repo.update_analysis_status(
                    analysis_id, "processing", 80.0, "Salvando resultados"
                )
                await websocket_manager.send_analysis_status(
                    analysis_id, "processing", 80.0, "Salvando resultados"
                )
                
                await repo.update_analysis_results(
                    analysis_id=analysis_id,
                    results=eda_results.get("results"),
                    summary=eda_results.get("summary"),
                    visualizations=eda_results.get("visualizations")
                )
                
                # 5. Finalizar
                await repo.update_analysis_status(
                    analysis_id=analysis_id,
                    status="completed",
                    progress=100.0,
                    message="Análise concluída com sucesso",
                    completed_at=datetime.utcnow()
                )
                
                # Notificar conclusão
                await websocket_manager.send_analysis_completed(
                    analysis_id, eda_results.get("summary")
                )
                
                logger.info(f"Análise concluída: {analysis_id}")
                break
                
        except Exception as e:
            logger.error(f"Erro na análise {analysis_id}: {e}")
            
            # Salvar erro no banco
            async for db_session in get_db_session():
                repo = AnalysisRepository(db_session)
                await repo.update_analysis_status(
                    analysis_id=analysis_id,
                    status="failed",
                    progress=0.0,
                    message=f"Erro: {str(e)}",
                    completed_at=datetime.utcnow()
                )
                break
            
            # Notificar erro
            await websocket_manager.send_analysis_error(analysis_id, str(e))
            
        finally:
            # Remover da lista de tarefas ativas
            self._active_tasks.pop(analysis_id, None)
    
    async def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Obter status da análise do banco de dados"""
        async for db_session in get_db_session():
            repo = AnalysisRepository(db_session)
            analysis = await repo.get_analysis(analysis_id)
            
            if not analysis:
                return None
            
            return {
                "analysis_id": analysis.id,
                "status": analysis.status,
                "progress": analysis.progress,
                "message": analysis.message,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
                "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
                "file_metadata": {
                    "file_key": analysis.file_key,
                    "file_size": analysis.file_size,
                    "file_format": analysis.file_format,
                    "rows_count": analysis.rows_count,
                    "columns_count": analysis.columns_count
                } if analysis.file_size else None
            }
    
    async def get_analysis_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Obter resultados da análise do banco de dados"""
        async for db_session in get_db_session():
            repo = AnalysisRepository(db_session)
            analysis = await repo.get_analysis(analysis_id)
            
            if not analysis or analysis.status != "completed":
                return None
            
            return {
                "analysis_id": analysis.id,
                "status": analysis.status,
                "results": analysis.results,
                "summary": analysis.summary,
                "visualizations": analysis.visualizations,
                "file_metadata": {
                    "file_key": analysis.file_key,
                    "file_size": analysis.file_size,
                    "file_format": analysis.file_format,
                    "rows_count": analysis.rows_count,
                    "columns_count": analysis.columns_count
                },
                "timestamps": {
                    "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                    "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
                    "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None
                }
            }
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Deletar análise (soft delete)"""
        async for db_session in get_db_session():
            repo = AnalysisRepository(db_session)
            
            # Cancelar tarefa se estiver ativa
            if analysis_id in self._active_tasks:
                self._active_tasks[analysis_id].cancel()
                del self._active_tasks[analysis_id]
            
            return await repo.delete_analysis(analysis_id, soft_delete=True)
    
    async def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str = None
    ) -> List[Dict[str, Any]]:
        """Listar análises do banco de dados"""
        async for db_session in get_db_session():
            repo = AnalysisRepository(db_session)
            analyses = await repo.list_analyses(limit, offset, status)
            
            return [
                {
                    "analysis_id": analysis.id,
                    "file_key": analysis.file_key,
                    "status": analysis.status,
                    "progress": analysis.progress,
                    "analysis_type": analysis.analysis_type,
                    "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                    "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
                    "file_metadata": {
                        "rows_count": analysis.rows_count,
                        "columns_count": analysis.columns_count,
                        "file_size": analysis.file_size
                    } if analysis.rows_count else None
                }
                for analysis in analyses
            ]
    
    async def _download_file_from_r2(self, file_key: str) -> bytes:
        """Baixar arquivo do Cloudflare R2"""
        # Para teste, simular download usando arquivo local
        if file_key.startswith("test/"):
            # Para testes, criar dados simulados
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            data = {
                'id': range(1, 101),
                'numeric_var1': np.random.normal(100, 15, 100),
                'numeric_var2': np.random.exponential(2, 100),
                'category_var1': np.random.choice(['A', 'B', 'C', 'D'], 100),
                'category_var2': np.random.choice(['X', 'Y', 'Z'], 100),
                'date_var': pd.date_range('2023-01-01', periods=100, freq='D'),
                'boolean_var': np.random.choice([True, False], 100)
            }
            
            df = pd.DataFrame(data)
            # Adicionar alguns valores nulos
            df.loc[::10, 'numeric_var1'] = np.nan
            df.loc[::15, 'category_var1'] = np.nan
            
            # Converter para CSV em bytes
            csv_content = df.to_csv(index=False)
            return csv_content.encode('utf-8')
        else:
            # Para produção, usar R2
            return await asyncio.to_thread(r2_service.download_file, file_key)
    
    async def _load_dataframe(self, file_content: bytes, file_key: str) -> pd.DataFrame:
        """Carregar DataFrame a partir do conteúdo do arquivo"""
        def _load():
            file_obj = io.BytesIO(file_content)
            
            if file_key.lower().endswith('.csv'):
                return pd.read_csv(file_obj)
            elif file_key.lower().endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_obj)
            else:
                # Tentar CSV como padrão
                return pd.read_csv(file_obj)
        
        return await asyncio.to_thread(_load)
    
    async def _run_eda_analysis(
        self,
        df: pd.DataFrame,
        analysis_type: str,
        options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Executar análise EDA"""
        return await asyncio.to_thread(
            self.eda_processor.process_dataframe,
            df,
            analysis_type,
            options or {}
        )
    
    async def cleanup_old_analyses(self, days_old: int = 7) -> int:
        """Limpar análises antigas"""
        async for db_session in get_db_session():
            repo = AnalysisRepository(db_session)
            return await repo.cleanup_old_analyses(days_old)


# Instância global do analisador persistente
persistent_data_analyzer = PersistentDataAnalyzer()