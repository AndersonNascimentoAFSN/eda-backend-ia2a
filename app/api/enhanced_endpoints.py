"""
Endpoints aprimorados com WebSocket e persistência
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.websocket import websocket_handler, websocket_manager
from app.services.persistent_analyzer import persistent_data_analyzer
from app.repositories.analysis_repository import AnalysisRepository
from app.models.responses import AnalysisStartRequest, AnalysisStartResponse

router = APIRouter(prefix="/api/v1", tags=["Enhanced Analysis"])


@router.post("/analysis/start-enhanced", response_model=AnalysisStartResponse)
async def start_enhanced_analysis(request: AnalysisStartRequest):
    """
    Iniciar análise aprimorada com persistência e WebSocket
    
    Este endpoint inicia uma análise com:
    - Persistência em banco de dados
    - Notificações WebSocket em tempo real
    - Melhor rastreabilidade e recuperação
    """
    try:
        analysis_id = await persistent_data_analyzer.start_analysis(
            file_key=request.file_key,
            analysis_type=request.analysis_type,
            options=request.options
        )
        
        return AnalysisStartResponse(
            analysis_id=analysis_id,
            status="pending",
            message="Análise iniciada com persistência e notificações WebSocket",
            websocket_url=f"/api/v1/analysis/websocket/{analysis_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao iniciar análise: {str(e)}")


@router.get("/analysis/status-enhanced/{analysis_id}")
async def get_enhanced_analysis_status(analysis_id: str):
    """
    Obter status da análise do banco de dados
    
    Retorna status detalhado incluindo:
    - Progresso atual
    - Timestamps de cada etapa
    - Metadados do arquivo
    - Informações de conexões WebSocket ativas
    """
    status = await persistent_data_analyzer.get_analysis_status(analysis_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Análise não encontrada")
    
    # Adicionar informações de WebSocket
    active_sessions = websocket_manager.get_active_sessions_for_analysis(analysis_id)
    status["websocket_info"] = {
        "active_sessions": len(active_sessions),
        "session_ids": active_sessions
    }
    
    return status


@router.get("/analysis/results-enhanced/{analysis_id}")
async def get_enhanced_analysis_results(analysis_id: str):
    """
    Obter resultados da análise do banco de dados
    
    Retorna resultados completos incluindo:
    - Dados da análise EDA
    - Visualizações
    - Resumos
    - Metadados completos
    """
    results = await persistent_data_analyzer.get_analysis_results(analysis_id)
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada ou ainda não foi concluída"
        )
    
    return results


@router.get("/analysis/list")
async def list_analyses(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None)
):
    """
    Listar análises com filtros
    
    Permite filtrar por:
    - Status da análise
    - Paginação
    """
    analyses = await persistent_data_analyzer.list_analyses(
        limit=limit,
        offset=offset,
        status=status
    )
    
    return {
        "analyses": analyses,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "count": len(analyses)
        }
    }


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Deletar análise (soft delete)
    
    Remove a análise do sistema mantendo histórico
    """
    success = await persistent_data_analyzer.delete_analysis(analysis_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Análise não encontrada")
    
    return {"message": "Análise removida com sucesso", "analysis_id": analysis_id}


@router.websocket("/analysis/websocket/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str, client_id: Optional[str] = None):
    """
    WebSocket para notificações em tempo real
    
    Conecte-se a este endpoint para receber:
    - Atualizações de status em tempo real
    - Notificações de progresso
    - Alertas de conclusão ou erro
    
    Mensagens enviadas:
    - connection_established: Confirmação de conexão
    - analysis_status: Atualizações de status
    - analysis_completed: Análise concluída
    - analysis_error: Erro na análise
    - heartbeat: Manter conexão viva
    """
    await websocket_handler.handle_connection(websocket, analysis_id, client_id)


@router.get("/analysis/websocket-info")
async def get_websocket_info():
    """
    Informações sobre conexões WebSocket ativas
    
    Útil para monitoramento e debugging
    """
    return {
        "total_connections": websocket_manager.get_total_connections(),
        "analyses_with_connections": websocket_manager.get_analyses_with_connections(),
        "active_connections_by_analysis": {
            analysis_id: len(websocket_manager.get_active_sessions_for_analysis(analysis_id))
            for analysis_id in websocket_manager.get_analyses_with_connections()
        }
    }


@router.post("/analysis/broadcast-test/{analysis_id}")
async def test_broadcast_message(analysis_id: str, message: Dict[str, Any]):
    """
    Teste de broadcast para uma análise (apenas para desenvolvimento)
    """
    sent_count = await websocket_manager.broadcast_to_analysis(analysis_id, {
        "type": "test_message",
        "data": message,
        "timestamp": "2024-01-01T00:00:00Z"
    })
    
    return {
        "message": "Mensagem enviada",
        "analysis_id": analysis_id,
        "recipients": sent_count
    }


@router.get("/analysis/summary-for-llm/{analysis_id}")
async def get_llm_summary(analysis_id: str):
    """
    Resumo otimizado para LLMs e agentes
    
    Retorna dados estruturados e resumidos especificamente
    formatados para consumo por agentes de IA
    """
    results = await persistent_data_analyzer.get_analysis_results(analysis_id)
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada ou ainda não foi concluída"
        )
    
    # Extrair informações mais relevantes para LLM
    summary = results.get("summary", {})
    file_metadata = results.get("file_metadata", {})
    
    llm_summary = {
        "analysis_id": analysis_id,
        "status": results["status"],
        "dataset_overview": {
            "total_records": summary.get("total_records", 0),
            "total_columns": summary.get("total_columns", 0),
            "numeric_columns": summary.get("numeric_columns", 0),
            "categorical_columns": summary.get("categorical_columns", 0),
            "missing_values": summary.get("missing_values_total", 0),
            "file_size_mb": round((file_metadata.get("file_size", 0) or 0) / 1024 / 1024, 2)
        },
        "data_quality": {
            "completeness": 100 - (summary.get("missing_values_total", 0) / summary.get("total_records", 1) * 100),
            "data_types_distribution": {
                "numeric": summary.get("numeric_columns", 0),
                "categorical": summary.get("categorical_columns", 0)
            }
        },
        "key_insights": [
            f"Dataset contains {summary.get('total_records', 0):,} records across {summary.get('total_columns', 0)} columns",
            f"Data completeness: {100 - (summary.get('missing_values_total', 0) / summary.get('total_records', 1) * 100):.1f}%",
            f"Mix of {summary.get('numeric_columns', 0)} numeric and {summary.get('categorical_columns', 0)} categorical variables"
        ],
        "available_analyses": [
            "Statistical summaries for numeric variables",
            "Distribution analysis",
            "Correlation analysis",
            "Missing value patterns"
        ],
        "timestamps": results.get("timestamps", {})
    }
    
    return llm_summary


@router.post("/maintenance/cleanup-old-analyses")
async def cleanup_old_analyses(days_old: int = Query(7, ge=1, le=30)):
    """
    Limpar análises antigas (endpoint de manutenção)
    
    Remove análises mais antigas que o número especificado de dias
    """
    cleaned_count = await persistent_data_analyzer.cleanup_old_analyses(days_old)
    
    return {
        "message": f"Limpeza concluída",
        "analyses_cleaned": cleaned_count,
        "days_threshold": days_old
    }


@router.get("/health/database")
async def health_check_database(db_session: AsyncSession = Depends(get_db_session)):
    """
    Health check do banco de dados
    """
    try:
        repo = AnalysisRepository(db_session)
        # Teste simples de conectividade
        analyses = await repo.list_analyses(limit=1)
        
        return {
            "status": "healthy",
            "database": "connected",
            "test_query": "successful"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )