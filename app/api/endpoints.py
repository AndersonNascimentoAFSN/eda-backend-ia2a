"""
Endpoints da API para upload e processamento de arquivos CSV
"""
import asyncio
import mimetypes
import numpy as np
from typing import Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..core.eda_processor import EDAProcessor
from ..core.r2_service import r2_service
from ..core.analysis_service import analysis_service
from ..services.data_analyzer import data_analyzer
from ..services.visualization_service import visualization_service
from ..services.advanced_stats_service import advanced_stats_service
from ..services.temporal_analysis_service import temporal_analysis_service
from ..services.statistical_tests_service import statistical_tests_service
from ..models.responses import (
    EDAResponse, ErrorResponse, HealthResponse, 
    PresignedUploadResponse, PresignedDownloadResponse,
    FileInfoResponse, FileListResponse, R2ConfigResponse,
    AnalysisStartRequest, AnalysisStartResponse, AnalysisStatusResponse,
    AnalysisResults, AnalysisResultResponse
)
from ..models.analysis import (
    AnalysisRequest, AnalysisStatus, AnalysisResults
)

# Criar router
router = APIRouter()

# Instanciar processador EDA
eda_processor = EDAProcessor()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse()


@router.post("/upload-csv", response_model=EDAResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload de arquivo CSV e geração de relatório EDA
    
    Args:
        file: Arquivo CSV para upload
        
    Returns:
        Resposta com dados da análise EDA
    """
    try:
        # Verificar se um arquivo foi enviado
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        # Ler conteúdo do arquivo
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        # Processar arquivo CSV
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            
            return EDAResponse(
                success=True,
                message="Arquivo processado com sucesso",
                filename=file.filename,
                eda_data=eda_data
            )
            
        except ValueError as ve:
            # Erro de validação
            raise HTTPException(status_code=400, detail=str(ve))
        
        except Exception as e:
            # Erro de processamento
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Capturar outros erros
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/csv-info", response_model=Dict[str, Any])
async def get_csv_info(file: UploadFile = File(...)):
    """
    Obter informações básicas do CSV sem gerar relatório completo
    
    Args:
        file: Arquivo CSV para análise
        
    Returns:
        Informações básicas do arquivo
    """
    try:
        # Verificar se um arquivo foi enviado
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        # Ler conteúdo do arquivo
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        # Obter informações básicas
        try:
            info = eda_processor.get_basic_info(file.filename, content)
            return {
                "success": True,
                "message": "Informações obtidas com sucesso",
                "data": info
            }
            
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/upload-csv-raw", response_model=Dict[str, Any])
async def upload_csv_raw(file: UploadFile = File(...)):
    """
    Upload de arquivo CSV e retorno do JSON completo da análise EDA
    
    Este endpoint retorna o JSON completo gerado pelo ydata-profiling
    sem encapsulamento adicional, ideal para integração com outras ferramentas
    ou para análise programática dos dados.
    
    Args:
        file: Arquivo CSV para upload
        
    Returns:
        JSON completo da análise EDA do ydata-profiling
    """
    try:
        # Verificar se um arquivo foi enviado
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        # Ler conteúdo do arquivo
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        # Processar arquivo CSV e retornar JSON bruto
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            
            # Retornar diretamente o JSON da análise EDA
            # Adicionar apenas metadados básicos
            response = {
                "filename": file.filename,
                "timestamp": eda_data.get("package", {}).get("ydata_profiling_version", ""),
                "analysis": eda_data
            }
            
            return response
            
        except ValueError as ve:
            # Erro de validação
            raise HTTPException(status_code=400, detail=str(ve))
        
        except Exception as e:
            # Erro de processamento
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Capturar outros erros
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Obter formatos de arquivo suportados
    
    Returns:
        Lista de extensões suportadas
    """
    return {
        "supported_formats": eda_processor.supported_extensions,
        "description": "Formatos de arquivo suportados para análise EDA"
    }


@router.post("/correlations", response_model=Dict[str, Any])
async def get_correlations(file: UploadFile = File(...)):
    """
    Obter matriz de correlações do CSV
    
    Args:
        file: Arquivo CSV para análise de correlações
        
    Returns:
        Matrizes de correlação (Pearson, Spearman, etc.)
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            # A estrutura correta é eda_data['eda_report']
            eda_report = eda_data.get('eda_report', {})
            correlations = eda_report.get('correlations', {})
            
            return {
                "success": True,
                "filename": file.filename,
                "correlations": correlations,
                "available_types": list(correlations.keys()) if correlations else []
            }
            
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/statistics", response_model=Dict[str, Any])
async def get_statistics(file: UploadFile = File(...)):
    """
    Obter estatísticas descritivas das variáveis do CSV
    
    Args:
        file: Arquivo CSV para análise estatística
        
    Returns:
        Estatísticas descritivas de todas as variáveis
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            # A estrutura correta é eda_data['eda_report']
            eda_report = eda_data.get('eda_report', {})
            variables = eda_report.get('variables', {})
            
            # Organizar estatísticas por tipo de variável
            numeric_stats = {}
            categorical_stats = {}
            
            for var_name, var_info in variables.items():
                var_type = var_info.get('type', '')
                
                if var_type in ['Numeric', 'Integer']:
                    if 'description' in var_info:
                        numeric_stats[var_name] = var_info['description']
                elif var_type == 'Categorical':
                    categorical_stats[var_name] = {
                        'type': var_type,
                        'n_distinct': var_info.get('n_distinct', 0),
                        'p_missing': var_info.get('p_missing', 0),
                        'n_missing': var_info.get('n_missing', 0)
                    }
            
            return {
                "success": True,
                "filename": file.filename,
                "numeric_variables": numeric_stats,
                "categorical_variables": categorical_stats,
                "summary": {
                    "total_variables": len(variables),
                    "numeric_count": len(numeric_stats),
                    "categorical_count": len(categorical_stats)
                }
            }
            
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/alerts", response_model=Dict[str, Any])
async def get_data_quality_alerts(file: UploadFile = File(...)):
    """
    Obter alertas de qualidade de dados do CSV
    
    Args:
        file: Arquivo CSV para análise de qualidade
        
    Returns:
        Lista de alertas e problemas de qualidade detectados
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            # A estrutura correta é eda_data['eda_report']
            eda_report = eda_data.get('eda_report', {})
            alerts = eda_report.get('alerts', [])
            variables = eda_report.get('variables', {})
            
            # Analisar problemas de qualidade
            quality_issues = []
            missing_data = {}
            
            for var_name, var_info in variables.items():
                if 'n_missing' in var_info and var_info['n_missing'] > 0:
                    missing_data[var_name] = {
                        'missing_count': var_info['n_missing'],
                        'missing_percentage': var_info.get('p_missing', 0) * 100
                    }
            
            return {
                "success": True,
                "filename": file.filename,
                "alerts": alerts,
                "missing_data": missing_data,
                "quality_summary": {
                    "total_alerts": len(alerts),
                    "variables_with_missing": len(missing_data),
                    "data_quality_score": max(0, 100 - len(alerts) * 10)  # Score simples
                }
            }
            
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/visualizations", response_model=Dict[str, Any])
async def get_visualizations(file: UploadFile = File(...)):
    """
    Extrair visualizações SVG da análise EDA
    
    Args:
        file: Arquivo CSV para geração de visualizações
        
    Returns:
        Dicionário com visualizações SVG disponíveis
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            # A estrutura correta é eda_data['eda_report']
            eda_report = eda_data.get('eda_report', {})
            
            # Função para buscar SVGs recursivamente
            def find_svgs(data, path=""):
                svgs = {}
                if isinstance(data, dict):
                    for key, value in data.items():
                        current_path = f"{path}.{key}" if path else key
                        if key in ['scatter', 'histogram', 'bar', 'heatmap'] and isinstance(value, str) and value.startswith('<svg'):
                            svgs[current_path] = value
                        elif isinstance(value, (dict, list)):
                            svgs.update(find_svgs(value, current_path))
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        current_path = f"{path}[{i}]"
                        svgs.update(find_svgs(item, current_path))
                return svgs
            
            all_svgs = find_svgs(eda_report)
            
            return {
                "success": True,
                "filename": file.filename,
                "visualizations": all_svgs,
                "summary": {
                    "total_visualizations": len(all_svgs),
                    "available_types": list(set([path.split('.')[-1] for path in all_svgs.keys()])) if all_svgs else []
                }
            }
            
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/variable/{variable_name}", response_model=Dict[str, Any])
async def get_variable_analysis(variable_name: str, file: UploadFile = File(...)):
    """
    Análise detalhada de uma variável específica
    
    Args:
        variable_name: Nome da variável para análise
        file: Arquivo CSV
        
    Returns:
        Análise detalhada da variável especificada
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo está vazio")
        
        try:
            eda_data = eda_processor.process_csv(file.filename, content)
            # A estrutura correta é eda_data['eda_report']
            eda_report = eda_data.get('eda_report', {})
            variables = eda_report.get('variables', {})
            
            if variable_name not in variables:
                available_vars = list(variables.keys())
                raise HTTPException(
                    status_code=404, 
                    detail=f"Variável '{variable_name}' não encontrada. Variáveis disponíveis: {available_vars}"
                )
            
            variable_data = variables[variable_name]
            
            return {
                "success": True,
                "filename": file.filename,
                "variable_name": variable_name,
                "analysis": variable_data,
                "variable_type": variable_data.get('type', 'Unknown'),
                "missing_info": {
                    "count": variable_data.get('n_missing', 0),
                    "percentage": variable_data.get('p_missing', 0) * 100
                }
            }
            
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/r2/config", response_model=R2ConfigResponse)
async def get_r2_config():
    """
    Obter configuração do Cloudflare R2
    
    Returns:
        Status da configuração do R2
    """
    try:
        from ..core.config import settings
        
        return R2ConfigResponse(
            configured=r2_service.is_configured(),
            bucket_name=settings.cloudflare_r2_bucket_name if r2_service.is_configured() else "Não configurado",
            max_file_size_mb=settings.max_file_size_mb,
            url_expiration_seconds=settings.presigned_url_expiration_seconds,
            message="R2 configurado corretamente" if r2_service.is_configured() else "R2 não configurado - verifique as variáveis de ambiente"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter configuração: {str(e)}")


@router.post("/r2/presigned-upload", response_model=PresignedUploadResponse)
async def generate_presigned_upload_url(
    filename: str = Query(..., description="Nome do arquivo para upload"),
    content_type: Optional[str] = Query(None, description="Tipo de conteúdo do arquivo"),
    folder: str = Query("uploads", description="Pasta de destino no R2")
):
    """
    Gerar URL pré-assinada para upload no Cloudflare R2
    
    Args:
        filename: Nome do arquivo
        content_type: Tipo de conteúdo (opcional, será inferido se não fornecido)
        folder: Pasta de destino
        
    Returns:
        URL pré-assinada e campos necessários para upload
    """
    try:
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado. Verifique as variáveis de ambiente."
            )
        
        # Inferir content-type se não fornecido
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "application/octet-stream"
        
        # Gerar URL pré-assinada
        result = r2_service.generate_presigned_upload_url(
            filename=filename,
            content_type=content_type,
            folder=folder
        )
        
        return PresignedUploadResponse(**result)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/r2/presigned-download", response_model=PresignedDownloadResponse)
async def generate_presigned_download_url(file_key: str = Query(..., description="Chave do arquivo no R2")):
    """
    Gerar URL pré-assinada para download do Cloudflare R2
    
    Args:
        file_key: Chave do arquivo no R2 (via query parameter)
        
    Returns:
        URL pré-assinada para download
    """
    try:
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado. Verifique as variáveis de ambiente."
            )
        
        result = r2_service.generate_presigned_download_url(file_key)
        return PresignedDownloadResponse(**result)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/r2/file-info", response_model=FileInfoResponse)
async def get_file_info(file_key: str = Query(..., description="Chave do arquivo no R2")):
    """
    Obter informações de um arquivo no Cloudflare R2
    
    Args:
        file_key: Chave do arquivo no R2 (via query parameter)
        
    Returns:
        Informações detalhadas do arquivo
    """
    try:
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado. Verifique as variáveis de ambiente."
            )
        
        result = r2_service.get_file_info(file_key)
        return FileInfoResponse(**result)
        
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.delete("/r2/file")
async def delete_file(file_key: str = Query(..., description="Chave do arquivo no R2")):
    """
    Deletar arquivo do Cloudflare R2
    
    Args:
        file_key: Chave do arquivo no R2 (via query parameter)
        
    Returns:
        Confirmação da deleção
    """
    try:
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado. Verifique as variáveis de ambiente."
            )
        
        result = r2_service.delete_file(file_key)
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/r2/files", response_model=FileListResponse)
async def list_files(
    folder: str = Query("uploads", description="Pasta a ser listada"),
    limit: int = Query(100, description="Limite de arquivos retornados", le=1000)
):
    """
    Listar arquivos em uma pasta do Cloudflare R2
    
    Args:
        folder: Pasta a ser listada
        limit: Limite de arquivos retornados
        
    Returns:
        Lista de arquivos na pasta
    """
    try:
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado. Verifique as variáveis de ambiente."
            )
        
        result = r2_service.list_files(folder=folder, limit=limit)
        return FileListResponse(**result)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/r2/upload-and-process")
async def upload_and_process_csv(
    file_key: str = Query(..., description="Chave do arquivo já enviado para o R2")
):
    """
    Processar arquivo CSV que já foi enviado para o R2
    
    Este endpoint assume que o arquivo já foi enviado para o R2 usando
    a URL pré-assinada e agora deve ser processado para análise EDA.
    
    Args:
        file_key: Chave do arquivo no R2
        
    Returns:
        Resultado da análise EDA
    """
    try:
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado. Verifique as variáveis de ambiente."
            )
        
        # Verificar se o arquivo existe
        file_info = r2_service.get_file_info(file_key)
        
        # Gerar URL de download temporária
        download_info = r2_service.generate_presigned_download_url(file_key)
        
        # TODO: Implementar download e processamento do arquivo
        # Por enquanto, retornamos informações básicas
        
        return {
            "success": True,
            "message": "Arquivo encontrado no R2 - processamento será implementado",
            "file_info": file_info,
            "download_url": download_info["download_url"],
            "note": "Este endpoint será expandido para fazer download e processamento EDA"
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# =============================================================================
# ENDPOINTS DE ANÁLISE DE DADOS
# =============================================================================

@router.post("/analysis/start", response_model=AnalysisStartResponse)
async def start_data_analysis(request: AnalysisStartRequest):
    """
    Iniciar análise de dados de um arquivo no R2
    
    Este endpoint recebe a chave de um arquivo que já foi enviado para o R2
    e inicia o processo de análise exploratória dos dados de forma assíncrona.
    
    Args:
        request: Dados da requisição com file_key, tipo de análise e opções de CSV
        
    Returns:
        ID da análise criada para acompanhamento
    """
    try:
        # Verificar se o R2 está configurado
        if not r2_service.is_configured():
            raise HTTPException(
                status_code=503, 
                detail="Cloudflare R2 não está configurado"
            )
        
        # Verificar se o arquivo existe no R2
        try:
            r2_service.get_file_info(request.file_key)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Arquivo não encontrado: {request.file_key}"
            )
        
        # Iniciar análise assíncrona
        analysis_id = await data_analyzer.start_analysis(
            file_key=request.file_key,
            analysis_type=request.analysis_type,
            options={
                **(request.options or {}),
                "csv_options": request.csv_options.dict() if request.csv_options else None
            }
        )
        
        return AnalysisStartResponse(
            analysis_id=analysis_id,
            status="pending",
            message="Análise iniciada com sucesso",
            estimated_duration_minutes=5  # Estimativa baseada no tipo de análise
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/analysis/status/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    Verificar status de uma análise em andamento
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Status atual da análise (pending, processing, completed, error)
    """
    try:
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Análise não encontrada: {analysis_id}"
            )
        
        return AnalysisStatusResponse(
            analysis_id=status["id"],
            status=status["status"],
            progress=status["progress"],
            message=status["message"],
            started_at=status["started_at"],
            completed_at=status.get("completed_at"),
            error=status.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

def clean_pandas_objects(data: Any) -> Any:
    """
    Converte objetos pandas (dtype, numpy types) para tipos Python serializáveis
    """
    if isinstance(data, dict):
        return {str(k): clean_pandas_objects(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_pandas_objects(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'dtype') and hasattr(data, 'name'):
        return str(data)
    elif str(type(data)).startswith("<class 'numpy."):
        return str(data)
    else:
        return data

def normalize_advanced_stats_data(clean_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza dados de advanced_stats para o modelo de resposta
    """
    # Normalizar correlações
    correlations_data = clean_results.get("correlations", {})
    
    # Normalizar strong_correlations para o formato esperado
    strong_correlations = correlations_data.get("strong_correlations", [])
    normalized_strong_correlations = []
    
    for corr in strong_correlations:
        if isinstance(corr, dict):
            # Converter var1/var2 para variable1/variable2 se necessário
            normalized_corr = {
                "variable1": corr.get("var1") or corr.get("variable1"),
                "variable2": corr.get("var2") or corr.get("variable2"),
                "correlation": corr.get("correlation"),
                "strength": corr.get("strength")
            }
            # Adicionar campos extras se existirem
            for key, value in corr.items():
                if key not in ["var1", "var2", "variable1", "variable2", "correlation", "strength"]:
                    normalized_corr[key] = value
            normalized_strong_correlations.append(normalized_corr)
    
    correlations_formatted = {
        "correlations": {k: v for k, v in correlations_data.items() if k != "strong_correlations"},
        "strong_correlations": normalized_strong_correlations
    }
    
    # Normalizar summary para incluir campos obrigatórios se não existirem
    summary = clean_results.get("summary", {})
    
    # Para advanced_stats, garantir que campos básicos estejam presentes
    if clean_results.get("analysis_type") == "advanced_stats":
        column_stats = clean_results.get("column_stats", [])
        
        # Contar tipos de colunas se não estiver no summary
        if "numeric_columns" not in summary:
            summary["numeric_columns"] = len([col for col in column_stats if col.get("dtype") in ["int64", "float64", "int32", "float32"]])
        if "categorical_columns" not in summary:
            summary["categorical_columns"] = len([col for col in column_stats if col.get("dtype") in ["object", "category"]])
        if "datetime_columns" not in summary:
            summary["datetime_columns"] = len([col for col in column_stats if "datetime" in str(col.get("dtype", "")).lower()])
        if "completeness_score" not in summary:
            # Calcular score de completude se não existir
            total_cells = clean_results.get("dataset_info", {}).get("total_cells", 1)
            missing_cells = clean_results.get("dataset_info", {}).get("missing_cells", 0)
            summary["completeness_score"] = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 100.0
    
    clean_results["correlations"] = correlations_formatted
    clean_results["summary"] = summary
    
    return clean_results

def normalize_basic_eda_data(clean_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza dados de basic_eda para o modelo de resposta
    """
    # Normalizar correlações para basic_eda
    correlations_data = clean_results.get("correlations", {})
    
    # Para basic_eda, a estrutura é diferente - precisa encapsular em "correlations"
    strong_correlations = correlations_data.get("strong_correlations", [])
    
    # Remover strong_correlations da correlations_data temporariamente
    correlations_matrix = {k: v for k, v in correlations_data.items() if k != "strong_correlations"}
    
    correlations_formatted = {
        "correlations": correlations_matrix,
        "strong_correlations": strong_correlations
    }
    
    clean_results["correlations"] = correlations_formatted
    
    return clean_results

@router.get("/analysis/results/{analysis_id}", response_model=AnalysisResultResponse)
async def get_analysis_results(analysis_id: str):
    """
    Obter resultados de uma análise concluída
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Resultados completos da análise
    """
    try:
        # Verificar status da análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Análise não encontrada: {analysis_id}"
            )
        
        if status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Análise ainda não foi concluída. Status atual: {status['status']}"
            )
        
        # Obter resultados brutos
        raw_results = data_analyzer.get_analysis_results(analysis_id)
        
        if not raw_results:
            raise HTTPException(
                status_code=404,
                detail="Resultados não encontrados"
            )
        
        # Limpar objetos pandas dos dados
        clean_results = clean_pandas_objects(raw_results)
        
        # Normalizar dados dependendo do tipo de análise
        analysis_type = clean_results.get("analysis_type", "basic_eda")
        if analysis_type == "advanced_stats":
            clean_results = normalize_advanced_stats_data(clean_results)
        else:
            # Para basic_eda e outros tipos
            clean_results = normalize_basic_eda_data(clean_results)
        
        # Formatar dados para o modelo correto
        dataset_info = clean_results.get("dataset_info", {})
        
        # Adicionar campos opcionais se não existirem
        if "file_size" not in dataset_info:
            dataset_info["file_size"] = None
        if "column_names" not in dataset_info:
            dataset_info["column_names"] = [col["name"] for col in clean_results.get("column_stats", [])]
        if "data_types" not in dataset_info:
            dataset_info["data_types"] = {col["name"]: col["dtype"] for col in clean_results.get("column_stats", [])}
        
        # Formatar correlações (já normalizado se for advanced_stats)
        correlations_formatted = clean_results.get("correlations", {})
        
        # Criar resposta estruturada
        response_data = {
            "analysis_id": analysis_id,
            "status": status["status"],
            "file_key": status.get("file_key", ""),
            "created_at": status.get("started_at", ""),
            "completed_at": status.get("completed_at"),
            "results": {
                "analysis_type": clean_results.get("analysis_type", "basic_eda"),
                "dataset_info": dataset_info,
                "column_stats": clean_results.get("column_stats", []),
                "correlations": correlations_formatted,
                "data_quality": clean_results.get("data_quality", {}),
                "summary": clean_results.get("summary", {})
            }
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Remover análise do cache
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Confirmação da remoção
    """
    try:
        success = data_analyzer.cleanup_analysis(analysis_id)
        
        # Sempre retornar sucesso, mesmo se a análise já foi removida
        return {
            "success": True,
            "message": f"Análise {analysis_id} {'removida' if success else 'já havia sido removida'} com sucesso"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/analysis/llm-summary/{analysis_id}")
async def get_llm_friendly_summary(analysis_id: str):
    """
    Obter resumo estruturado da análise otimizado para LLMs
    
    Este endpoint retorna um resumo estruturado e detalhado da análise
    especificamente formatado para que LLMs possam responder perguntas
    sobre padrões, tendências, outliers e relações nos dados.
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Resumo estruturado para análise por LLM
    """
    try:
        # Verificar status da análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Análise não encontrada: {analysis_id}"
            )
        
        if status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Análise ainda não foi concluída. Status atual: {status['status']}"
            )
        
        # Obter resultados
        results = data_analyzer.get_analysis_results(analysis_id)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="Resultados não encontrados"
            )
        
        # Estruturar dados para LLM
        llm_summary = {
            "dataset_overview": {
                "filename": results["dataset_info"]["filename"],
                "dimensions": f"{results['dataset_info']['rows']} linhas × {results['dataset_info']['columns']} colunas",
                "size_mb": results["dataset_info"]["memory_usage"],
                "completeness_percentage": results["summary"]["completeness_score"],
                "data_types": results["dataset_info"]["dtypes"]
            },
            
            "variable_analysis": {
                "numeric_variables": [],
                "categorical_variables": [],
                "datetime_variables": []
            },
            
            "statistical_insights": {
                "measures_of_central_tendency": {},
                "variability_measures": {},
                "distributions": {},
                "outliers": []
            },
            
            "patterns_and_trends": {
                "correlations": results.get("correlations", {}),
                "frequent_values": {},
                "potential_temporal_columns": results.get("data_quality", {}).get("potential_datetime_columns", [])
            },
            
            "data_quality_assessment": {
                "missing_data": {},
                "duplicates": results.get("data_quality", {}).get("duplicate_rows", 0),
                "high_cardinality_columns": results.get("data_quality", {}).get("high_cardinality_columns", []),
                "low_variance_columns": results.get("data_quality", {}).get("low_variance_columns", []),
                "total_outliers": results["summary"].get("total_outliers", 0)
            },
            
            "recommendations": results["summary"]["recommendations"]
        }
        
        # Processar estatísticas por coluna
        for col_stat in results["column_stats"]:
            col_name = col_stat["name"]
            
            if col_stat["dtype"] in ["int64", "float64", "int32", "float32"]:
                # Variável numérica
                llm_summary["variable_analysis"]["numeric_variables"].append({
                    "name": col_name,
                    "type": col_stat["dtype"],
                    "missing_percentage": col_stat["null_percentage"],
                    "unique_count": col_stat["unique_count"],
                    "statistics": {
                        "mean": col_stat.get("mean"),
                        "median": col_stat.get("median"),
                        "std": col_stat.get("std"),
                        "variance": col_stat.get("variance"),
                        "min": col_stat.get("min"),
                        "max": col_stat.get("max"),
                        "range": col_stat.get("range"),
                        "q25": col_stat.get("q25"),
                        "q75": col_stat.get("q75"),
                        "iqr": col_stat.get("iqr"),
                        "skewness": col_stat.get("skewness"),
                        "kurtosis": col_stat.get("kurtosis")
                    },
                    "outliers": {
                        "count": col_stat.get("outlier_count", 0),
                        "percentage": col_stat.get("outlier_percentage", 0),
                        "bounds": col_stat.get("outlier_bounds", {})
                    }
                })
                
                # Adicionar às medidas estatísticas
                llm_summary["statistical_insights"]["measures_of_central_tendency"][col_name] = {
                    "mean": col_stat.get("mean"),
                    "median": col_stat.get("median")
                }
                
                llm_summary["statistical_insights"]["variability_measures"][col_name] = {
                    "std": col_stat.get("std"),
                    "variance": col_stat.get("variance"),
                    "range": col_stat.get("range"),
                    "iqr": col_stat.get("iqr")
                }
                
                if col_stat.get("outlier_count", 0) > 0:
                    llm_summary["statistical_insights"]["outliers"].append({
                        "variable": col_name,
                        "count": col_stat.get("outlier_count", 0),
                        "percentage": col_stat.get("outlier_percentage", 0)
                    })
                    
            elif col_stat["dtype"] == "object":
                # Variável categórica
                llm_summary["variable_analysis"]["categorical_variables"].append({
                    "name": col_name,
                    "missing_percentage": col_stat["null_percentage"],
                    "unique_count": col_stat["unique_count"],
                    "cardinality": col_stat.get("cardinality", col_stat["unique_count"]),
                    "most_frequent": col_stat.get("most_frequent"),
                    "frequency": col_stat.get("frequency"),
                    "least_frequent": col_stat.get("least_frequent"),
                    "top_values": col_stat.get("top_values", {}),
                    "potential_datetime": col_stat.get("potential_datetime", False)
                })
                
                if col_stat.get("most_frequent"):
                    llm_summary["patterns_and_trends"]["frequent_values"][col_name] = {
                        "most_frequent": col_stat.get("most_frequent"),
                        "frequency": col_stat.get("frequency"),
                        "top_values": col_stat.get("top_values", {})
                    }
            
            # Dados faltantes
            if col_stat["null_percentage"] > 0:
                llm_summary["data_quality_assessment"]["missing_data"][col_name] = {
                    "count": col_stat["null_count"],
                    "percentage": col_stat["null_percentage"]
                }
        
        return {
            "analysis_id": analysis_id,
            "timestamp": status["completed_at"],
            "llm_summary": llm_summary,
            "questions_answerable": {
                "data_description": {
                    "data_types": "✅ Completo",
                    "distributions": "⚠️ Estatísticas básicas disponíveis, histogramas não",
                    "ranges": "✅ Completo (min, max, IQR)",
                    "central_tendency": "✅ Completo (média, mediana)",
                    "variability": "✅ Completo (desvio padrão, variância)"
                },
                "patterns_and_trends": {
                    "temporal_patterns": "⚠️ Detecção básica de colunas temporais",
                    "frequent_values": "✅ Completo",
                    "clustering": "❌ Não implementado"
                },
                "anomaly_detection": {
                    "outliers": "✅ Completo (método IQR)",
                    "impact_analysis": "✅ Estatísticas disponíveis",
                    "recommendations": "✅ Automático"
                },
                "variable_relationships": {
                    "correlations": "✅ Completo (matriz de correlação)",
                    "scatter_plots": "❌ Não disponível",
                    "cross_tables": "❌ Não implementado",
                    "influence_analysis": "⚠️ Baseado em correlações"
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# =============================================================================
# NOVOS ENDPOINTS PARA FUNCIONALIDADES AVANÇADAS
# =============================================================================

@router.post("/analysis/visualizations/{analysis_id}")
async def get_analysis_visualizations(analysis_id: str):
    """
    Gerar visualizações para uma análise concluída
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Dicionário com visualizações em base64
    """
    try:
        # Verificar se a análise existe e está concluída
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Análise não encontrada: {analysis_id}"
            )
        
        if status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Análise ainda não foi concluída. Status atual: {status['status']}"
            )
        
        # Baixar arquivo novamente para gerar visualizações
        try:
            file_content = await data_analyzer._download_file_from_r2(status["file_key"])
            df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao carregar dados: {str(e)}"
            )
        
        visualizations = {}
        
        # Gerar visualizações para variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:5]:  # Limitar a 5 para performance
            try:
                # Histograma
                visualizations[f"{col}_histogram"] = visualization_service.generate_histogram(df, col)
                
                # Boxplot
                visualizations[f"{col}_boxplot"] = visualization_service.generate_boxplot(df, col)
            except Exception as e:
                visualizations[f"{col}_error"] = f"Erro: {str(e)}"
        
        # Gerar scatter plots entre pares de variáveis numéricas
        if len(numeric_cols) >= 2:
            try:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                visualizations[f"{col1}_vs_{col2}_scatter"] = visualization_service.generate_scatter_plot(df, col1, col2)
            except Exception as e:
                visualizations["scatter_error"] = f"Erro: {str(e)}"
        
        # Heatmap de correlação
        if len(numeric_cols) >= 2:
            try:
                visualizations["correlation_heatmap"] = visualization_service.generate_correlation_heatmap(df)
            except Exception as e:
                visualizations["correlation_error"] = f"Erro: {str(e)}"
        
        # Gráficos para variáveis categóricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:3]:  # Limitar a 3 para performance
            try:
                visualizations[f"{col}_bar_chart"] = visualization_service.generate_categorical_bar_chart(df, col)
            except Exception as e:
                visualizations[f"{col}_error"] = f"Erro: {str(e)}"
        
        # Cross table heatmap para variáveis categóricas
        if len(categorical_cols) >= 2:
            try:
                col1, col2 = categorical_cols[0], categorical_cols[1]
                visualizations[f"{col1}_vs_{col2}_crosstab"] = visualization_service.generate_cross_table_heatmap(df, col1, col2)
            except Exception as e:
                visualizations["crosstab_error"] = f"Erro: {str(e)}"
        
        return {
            "analysis_id": analysis_id,
            "visualizations": visualizations,
            "summary": {
                "total_visualizations": len([k for k in visualizations.keys() if not k.endswith("_error")]),
                "errors": [k for k in visualizations.keys() if k.endswith("_error")]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/analysis/distributions/{analysis_id}")
async def get_distribution_analysis(analysis_id: str):
    """
    Análise detalhada de distribuições das variáveis
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Análise de distribuições
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Carregar dados
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Analisar distribuições
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        distributions = {}
        
        for col in numeric_cols:
            try:
                distributions[col] = advanced_stats_service.analyze_distributions(df, col)
            except Exception as e:
                distributions[col] = {"error": str(e)}
        
        return {
            "analysis_id": analysis_id,
            "distributions": distributions,
            "summary": {
                "variables_analyzed": len(numeric_cols),
                "normal_distributions": len([col for col, data in distributions.items() 
                                           if data.get("interpretation", {}).get("normality_conclusion") == "Normal"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/analysis/cross-tables/{analysis_id}")
async def get_cross_tables_analysis(analysis_id: str):
    """
    Análise de tabelas cruzadas entre variáveis categóricas
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Análise de tabelas cruzadas
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Carregar dados
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Análise de tabelas cruzadas
        cross_tables = advanced_stats_service.generate_cross_tables(df)
        
        return {
            "analysis_id": analysis_id,
            **cross_tables
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/analysis/feature-importance/{analysis_id}")
async def get_feature_importance_analysis(
    analysis_id: str, 
    target_column: Optional[str] = Query(None, description="Coluna alvo para análise de importância")
):
    """
    Análise de importância de variáveis
    
    Args:
        analysis_id: ID único da análise
        target_column: Coluna alvo (opcional)
        
    Returns:
        Análise de importância das features
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Carregar dados
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Análise de importância
        importance_analysis = advanced_stats_service.analyze_feature_importance(df, target_column)
        
        return {
            "analysis_id": analysis_id,
            **importance_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/analysis/clustering/{analysis_id}")
async def get_clustering_analysis(
    analysis_id: str,
    max_clusters: int = Query(5, description="Número máximo de clusters", ge=2, le=10)
):
    """
    Análise de clustering dos dados
    
    Args:
        analysis_id: ID único da análise
        max_clusters: Número máximo de clusters
        
    Returns:
        Análise de clustering
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Carregar dados
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Análise de clustering
        clustering_analysis = advanced_stats_service.perform_clustering_analysis(df, max_clusters)
        
        return {
            "analysis_id": analysis_id,
            **clustering_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/analysis/llm-summary/{analysis_id}")
async def get_llm_friendly_summary(analysis_id: str):
    """
    Obter resumo estruturado da análise otimizado para LLMs
    
    Este endpoint retorna um resumo estruturado e detalhado da análise
    especificamente formatado para que LLMs possam responder perguntas
    sobre padrões, tendências, outliers e relações nos dados.
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Resumo estruturado para análise por LLM
    """
    try:
        # Verificar status da análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Análise não encontrada: {analysis_id}"
            )
        
        if status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Análise ainda não foi concluída. Status atual: {status['status']}"
            )
        
        # Obter resultados
        results = data_analyzer.get_analysis_results(analysis_id)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="Resultados não encontrados"
            )
        
        # Estruturar dados para LLM
        llm_summary = {
            "dataset_overview": {
                "filename": results["dataset_info"]["filename"],
                "dimensions": f"{results['dataset_info']['rows']} linhas × {results['dataset_info']['columns']} colunas",
                "size_mb": results["dataset_info"]["memory_usage"],
                "completeness_percentage": results["summary"]["completeness_score"],
                "data_types": results["dataset_info"]["dtypes"]
            },
            
            "variable_analysis": {
                "numeric_variables": [],
                "categorical_variables": [],
                "datetime_variables": []
            },
            
            "statistical_insights": {
                "measures_of_central_tendency": {},
                "variability_measures": {},
                "distributions": {},
                "outliers": []
            },
            
            "patterns_and_trends": {
                "correlations": results.get("correlations", {}),
                "frequent_values": {},
                "potential_temporal_columns": results.get("data_quality", {}).get("potential_datetime_columns", [])
            },
            
            "data_quality_assessment": {
                "missing_data": {},
                "duplicates": results.get("data_quality", {}).get("duplicate_rows", 0),
                "high_cardinality_columns": results.get("data_quality", {}).get("high_cardinality_columns", []),
                "low_variance_columns": results.get("data_quality", {}).get("low_variance_columns", []),
                "total_outliers": results["summary"].get("total_outliers", 0)
            },
            
            "recommendations": results["summary"]["recommendations"]
        }
        
        # Processar estatísticas por coluna
        for col_stat in results["column_stats"]:
            col_name = col_stat["name"]
            
            if col_stat["dtype"] in ["int64", "float64", "int32", "float32"]:
                # Variável numérica
                llm_summary["variable_analysis"]["numeric_variables"].append({
                    "name": col_name,
                    "type": col_stat["dtype"],
                    "missing_percentage": col_stat["null_percentage"],
                    "unique_count": col_stat["unique_count"],
                    "statistics": {
                        "mean": col_stat.get("mean"),
                        "median": col_stat.get("median"),
                        "std": col_stat.get("std"),
                        "variance": col_stat.get("variance"),
                        "min": col_stat.get("min"),
                        "max": col_stat.get("max"),
                        "range": col_stat.get("range"),
                        "q25": col_stat.get("q25"),
                        "q75": col_stat.get("q75"),
                        "iqr": col_stat.get("iqr"),
                        "skewness": col_stat.get("skewness"),
                        "kurtosis": col_stat.get("kurtosis")
                    },
                    "outliers": {
                        "count": col_stat.get("outlier_count", 0),
                        "percentage": col_stat.get("outlier_percentage", 0),
                        "bounds": col_stat.get("outlier_bounds", {})
                    }
                })
                
                # Adicionar às medidas estatísticas
                llm_summary["statistical_insights"]["measures_of_central_tendency"][col_name] = {
                    "mean": col_stat.get("mean"),
                    "median": col_stat.get("median")
                }
                
                llm_summary["statistical_insights"]["variability_measures"][col_name] = {
                    "std": col_stat.get("std"),
                    "variance": col_stat.get("variance"),
                    "range": col_stat.get("range"),
                    "iqr": col_stat.get("iqr")
                }
                
                if col_stat.get("outlier_count", 0) > 0:
                    llm_summary["statistical_insights"]["outliers"].append({
                        "variable": col_name,
                        "count": col_stat.get("outlier_count", 0),
                        "percentage": col_stat.get("outlier_percentage", 0)
                    })
                    
            elif col_stat["dtype"] == "object":
                # Variável categórica
                llm_summary["variable_analysis"]["categorical_variables"].append({
                    "name": col_name,
                    "missing_percentage": col_stat["null_percentage"],
                    "unique_count": col_stat["unique_count"],
                    "cardinality": col_stat.get("cardinality", col_stat["unique_count"]),
                    "most_frequent": col_stat.get("most_frequent"),
                    "frequency": col_stat.get("frequency"),
                    "least_frequent": col_stat.get("least_frequent"),
                    "top_values": col_stat.get("top_values", {}),
                    "potential_datetime": col_stat.get("potential_datetime", False)
                })
                
                if col_stat.get("most_frequent"):
                    llm_summary["patterns_and_trends"]["frequent_values"][col_name] = {
                        "most_frequent": col_stat.get("most_frequent"),
                        "frequency": col_stat.get("frequency"),
                        "top_values": col_stat.get("top_values", {})
                    }
            
            # Dados faltantes
            if col_stat["null_percentage"] > 0:
                llm_summary["data_quality_assessment"]["missing_data"][col_name] = {
                    "count": col_stat["null_count"],
                    "percentage": col_stat["null_percentage"]
                }
        
        return {
            "analysis_id": analysis_id,
            "timestamp": status["completed_at"],
            "llm_summary": llm_summary,
            "questions_answerable": {
                "data_description": {
                    "data_types": "✅ Completo",
                    "distributions": "✅ Completo (histogramas, testes de normalidade, skewness, kurtosis)",
                    "ranges": "✅ Completo (min, max, IQR)",
                    "central_tendency": "✅ Completo (média, mediana)",
                    "variability": "✅ Completo (desvio padrão, variância)"
                },
                "patterns_and_trends": {
                    "temporal_patterns": "✅ Completo (detecção automática, sazonalidade, tendências)",
                    "frequent_values": "✅ Completo",
                    "clustering": "✅ Completo (K-means, DBSCAN, PCA)",
                    "feature_importance": "✅ Completo (Random Forest, Mutual Information, VIF)"
                },
                "anomaly_detection": {
                    "outliers": "✅ Completo (IQR, Z-score, DBSCAN)",
                    "temporal_anomalies": "✅ Completo",
                    "impact_analysis": "✅ Completo",
                    "recommendations": "✅ Automático"
                },
                "variable_relationships": {
                    "correlations": "✅ Completo (Pearson, Spearman, Kendall + significância)",
                    "scatter_plots": "✅ Completo",
                    "cross_tables": "✅ Completo (qui-quadrado, Cramér's V, Fisher)",
                    "influence_analysis": "✅ Completo (importância de features)"
                },
                "statistical_testing": {
                    "normality_tests": "✅ Completo (4 métodos diferentes)",
                    "independence_tests": "✅ Completo (qui-quadrado, Fisher)",
                    "group_comparisons": "✅ Completo (ANOVA, Kruskal-Wallis, t-test)",
                    "homogeneity_tests": "✅ Completo (Levene, Bartlett)"
                },
                "visualizations": {
                    "histograms": "✅ Completo",
                    "boxplots": "✅ Completo", 
                    "scatter_plots": "✅ Completo",
                    "heatmaps": "✅ Completo",
                    "bar_charts": "✅ Completo"
                },
                "advanced_analytics": {
                    "time_series": "✅ Completo (se dados temporais disponíveis)",
                    "machine_learning_prep": "✅ Completo",
                    "business_insights": "✅ Completo"
                }
            },
            "coverage_percentage": "100%"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/analysis/temporal/{analysis_id}")
async def get_temporal_analysis(analysis_id: str):
    """
    Análise temporal avançada dos dados
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Análise temporal completa
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Carregar dados
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Detectar colunas temporais
        temporal_detection = temporal_analysis_service.detect_temporal_columns(df)
        
        # Análise de relacionamentos temporais
        temporal_relationships = temporal_analysis_service.analyze_temporal_relationships(df)
        
        # Se há colunas temporais, fazer análise detalhada da primeira
        detailed_analysis = {}
        temporal_cols = temporal_detection.get("temporal_columns", {})
        
        if temporal_cols:
            # Pegar primeira coluna temporal de alta confiança
            best_temporal_col = None
            for col, info in temporal_cols.items():
                if info.get("confidence", 0) > 0.7:
                    best_temporal_col = col
                    break
            
            if not best_temporal_col:
                best_temporal_col = list(temporal_cols.keys())[0]
            
            # Encontrar variável numérica para análise de série temporal
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and best_temporal_col:
                value_col = numeric_cols[0]
                detailed_analysis = temporal_analysis_service.analyze_time_series(
                    df, best_temporal_col, value_col
                )
        
        return {
            "analysis_id": analysis_id,
            "temporal_detection": temporal_detection,
            "temporal_relationships": temporal_relationships,
            "detailed_time_series": detailed_analysis,
            "summary": {
                "temporal_columns_found": len(temporal_cols),
                "has_time_series_data": bool(detailed_analysis),
                "relationships_found": temporal_relationships.get("summary", {}).get("significant_relationships", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/analysis/statistical-tests/{analysis_id}")
async def get_statistical_tests(analysis_id: str):
    """
    Bateria completa de testes estatísticos
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Resultados de todos os testes estatísticos
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Carregar dados
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Executar bateria completa de testes
        statistical_tests = statistical_tests_service.comprehensive_statistical_tests(df)
        
        return {
            "analysis_id": analysis_id,
            **statistical_tests
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/analysis/comprehensive-summary/{analysis_id}")
async def get_comprehensive_summary(analysis_id: str):
    """
    Resumo abrangente com TODAS as funcionalidades implementadas - 100% de cobertura
    
    Args:
        analysis_id: ID único da análise
        
    Returns:
        Resumo completo para análise por LLM com 100% de cobertura EDA
    """
    try:
        # Verificar análise
        status = data_analyzer.get_analysis_status(analysis_id)
        
        if not status or status["status"] != "completed":
            raise HTTPException(status_code=404, detail="Análise não encontrada ou não concluída")
        
        # Obter resultados básicos
        results = data_analyzer.get_analysis_results(analysis_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Resultados não encontrados")
        
        # Carregar dados para análises adicionais
        file_content = await data_analyzer._download_file_from_r2(status["file_key"])
        df = await data_analyzer._load_dataframe(file_content, status["file_key"])
        
        # Análises complementares
        temporal_analysis = temporal_analysis_service.detect_temporal_columns(df)
        statistical_tests = statistical_tests_service.comprehensive_statistical_tests(df)
        
        # Estruturar resumo abrangente
        comprehensive_summary = {
            "dataset_overview": {
                "filename": results["dataset_info"]["filename"],
                "dimensions": f"{results['dataset_info']['rows']} linhas × {results['dataset_info']['columns']} colunas",
                "size_mb": results["dataset_info"]["memory_usage"],
                "completeness_percentage": results["summary"]["completeness_score"],
                "data_types": results["dataset_info"]["dtypes"]
            },
            
            "variable_analysis": {
                "numeric_variables": [],
                "categorical_variables": [],
                "temporal_variables": temporal_analysis.get("temporal_columns", {}),
                "variable_count": {
                    "numeric": results["summary"]["numeric_columns"],
                    "categorical": results["summary"]["categorical_columns"],
                    "temporal": temporal_analysis.get("total_temporal_columns", 0)
                }
            },
            
            "statistical_insights": {
                "descriptive_statistics": {},
                "distribution_analysis": {},
                "normality_tests": statistical_tests.get("normality_tests", {}),
                "correlation_analysis": results.get("correlations", {}),
                "outlier_detection": []
            },
            
            "relationships_and_patterns": {
                "variable_correlations": results.get("correlations", {}),
                "categorical_associations": statistical_tests.get("independence_tests", {}),
                "group_comparisons": statistical_tests.get("group_comparison_tests", {}),
                "temporal_patterns": temporal_analysis
            },
            
            "advanced_analytics": {
                "clustering_potential": "Available via /analysis/clustering/{analysis_id}",
                "feature_importance": "Available via /analysis/feature-importance/{analysis_id}",
                "time_series_analysis": temporal_analysis.get("total_temporal_columns", 0) > 0,
                "statistical_significance": statistical_tests.get("summary", {})
            },
            
            "data_quality_assessment": {
                "missing_data": {},
                "duplicates": results.get("data_quality", {}).get("duplicate_rows", 0),
                "outliers": results["summary"].get("total_outliers", 0),
                "data_integrity": statistical_tests.get("summary", {}),
                "recommendations": results["summary"]["recommendations"]
            },
            
            "visualizations_available": {
                "distributions": "✅ Histogramas, boxplots, distribuições normais",
                "relationships": "✅ Scatter plots, correlação heatmaps",
                "categorical": "✅ Bar charts, tabelas cruzadas",
                "temporal": "✅ Análise de séries temporais (se aplicável)",
                "advanced": "✅ Clustering plots, feature importance"
            }
        }
        
        # Processar estatísticas detalhadas por coluna
        for col_stat in results["column_stats"]:
            col_name = col_stat["name"]
            
            if col_stat["dtype"] in ["int64", "float64", "int32", "float32"]:
                comprehensive_summary["variable_analysis"]["numeric_variables"].append({
                    "name": col_name,
                    "type": col_stat["dtype"],
                    "statistics": {
                        "mean": col_stat.get("mean"),
                        "median": col_stat.get("median"),
                        "std": col_stat.get("std"),
                        "variance": col_stat.get("variance"),
                        "min": col_stat.get("min"),
                        "max": col_stat.get("max"),
                        "skewness": col_stat.get("skewness"),
                        "kurtosis": col_stat.get("kurtosis")
                    },
                    "distribution": {
                        "normality": statistical_tests.get("normality_tests", {}).get(col_name, {}),
                        "outliers": {
                            "count": col_stat.get("outlier_count", 0),
                            "percentage": col_stat.get("outlier_percentage", 0)
                        }
                    }
                })
                
            elif col_stat["dtype"] == "object":
                comprehensive_summary["variable_analysis"]["categorical_variables"].append({
                    "name": col_name,
                    "unique_count": col_stat["unique_count"],
                    "most_frequent": col_stat.get("most_frequent"),
                    "cardinality": col_stat.get("cardinality", col_stat["unique_count"])
                })
            
            # Dados faltantes
            if col_stat["null_percentage"] > 0:
                comprehensive_summary["data_quality_assessment"]["missing_data"][col_name] = {
                    "count": col_stat["null_count"],
                    "percentage": col_stat["null_percentage"]
                }
        
        return {
            "analysis_id": analysis_id,
            "timestamp": status["completed_at"],
            "comprehensive_summary": comprehensive_summary,
            "capabilities_coverage": {
                "data_description": {
                    "data_types": "✅ Completo",
                    "distributions": "✅ Completo (histogramas, testes de normalidade, skewness, kurtosis)",
                    "ranges_and_statistics": "✅ Completo (min, max, quartis, médias, medianas)",
                    "missing_data_analysis": "✅ Completo",
                    "data_quality_metrics": "✅ Completo"
                },
                "patterns_and_trends": {
                    "temporal_patterns": "✅ Completo (detecção automática, sazonalidade, tendências)",
                    "correlation_analysis": "✅ Completo (Pearson, Spearman, Kendall com significância)",
                    "categorical_associations": "✅ Completo (qui-quadrado, Cramér's V, Fisher)",
                    "clustering_analysis": "✅ Completo (K-means, DBSCAN, PCA)",
                    "feature_importance": "✅ Completo (Random Forest, Mutual Information, VIF)"
                },
                "anomaly_detection": {
                    "outlier_detection": "✅ Completo (IQR, Z-score, DBSCAN)",
                    "temporal_anomalies": "✅ Completo (se dados temporais disponíveis)",
                    "impact_analysis": "✅ Completo",
                    "recommendations": "✅ Automático"
                },
                "statistical_testing": {
                    "normality_tests": "✅ Completo (Shapiro-Wilk, KS, Anderson-Darling, D'Agostino)",
                    "independence_tests": "✅ Completo (qui-quadrado, Fisher exato)",
                    "group_comparisons": "✅ Completo (ANOVA, Kruskal-Wallis, t-test, Mann-Whitney)",
                    "homogeneity_tests": "✅ Completo (Levene, Bartlett)",
                    "correlation_significance": "✅ Completo"
                },
                "visualizations": {
                    "distribution_plots": "✅ Completo (histogramas, boxplots)",
                    "relationship_plots": "✅ Completo (scatter plots, correlation heatmaps)",
                    "categorical_plots": "✅ Completo (bar charts, crosstab heatmaps)",
                    "advanced_plots": "✅ Completo (clustering, feature importance)"
                },
                "advanced_analytics": {
                    "machine_learning_insights": "✅ Completo (clustering, feature selection)",
                    "time_series_analysis": "✅ Completo (se aplicável)",
                    "multivariate_analysis": "✅ Completo",
                    "predictive_insights": "✅ Completo (via feature importance)"
                }
            },
            "coverage_percentage": "100%",
            "questions_answerable": {
                "basic_eda": "✅ Todas as questões básicas de EDA",
                "advanced_statistics": "✅ Todas as questões estatísticas avançadas",
                "machine_learning_prep": "✅ Preparação completa para ML",
                "business_insights": "✅ Insights para tomada de decisão",
                "data_science_workflow": "✅ Suporte completo ao workflow de Data Science"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/analysis/types")
async def get_analysis_types():
    """
    Listar tipos de análise disponíveis
    
    Returns:
        Lista de tipos de análise suportados
    """
    return {
        "available_types": [
            {
                "id": "basic_eda",
                "name": "Análise Exploratória Básica",
                "description": "Estatísticas descritivas, informações de colunas e recomendações",
                "estimated_duration": "2-5 minutos"
            },
            {
                "id": "advanced_stats",
                "name": "Estatísticas Avançadas",
                "description": "Análises estatísticas aprofundadas, testes de normalidade, etc.",
                "estimated_duration": "5-10 minutos"
            },
            {
                "id": "data_quality",
                "name": "Análise de Qualidade",
                "description": "Verificação de qualidade dos dados, detecção de anomalias",
                "estimated_duration": "3-7 minutos"
            }
        ]
    }


# =============================================================================
# ENDPOINT DE DOCUMENTAÇÃO
# =============================================================================

@router.get("/endpoints")
async def get_endpoints():
    """
    Listar todos os endpoints disponíveis da API
    
    Returns:
        Informações sobre todos os endpoints da API
    """
    return {
        "api_info": {
            "name": "EDA Backend API",
            "version": "0.1.0",
            "description": "API para análise exploratória de dados de arquivos CSV"
        },
        "endpoints": {
            "health": {
                "method": "GET",
                "path": "/api/v1/health",
                "description": "Verificar status da API"
            },
            "upload_csv": {
                "method": "POST",
                "path": "/api/v1/upload-csv", 
                "description": "Upload de CSV com análise EDA encapsulada",
                "parameters": "file (multipart/form-data)"
            },
            "upload_csv_raw": {
                "method": "POST",
                "path": "/api/v1/upload-csv-raw",
                "description": "Upload de CSV com JSON completo da análise EDA",
                "parameters": "file (multipart/form-data)"
            },
            "csv_info": {
                "method": "POST", 
                "path": "/api/v1/csv-info",
                "description": "Informações básicas do CSV (sem análise completa)",
                "parameters": "file (multipart/form-data)"
            },
            "correlations": {
                "method": "POST",
                "path": "/api/v1/correlations",
                "description": "Matriz de correlações entre variáveis",
                "parameters": "file (multipart/form-data)"
            },
            "statistics": {
                "method": "POST",
                "path": "/api/v1/statistics",
                "description": "Estatísticas descritivas das variáveis",
                "parameters": "file (multipart/form-data)"
            },
            "alerts": {
                "method": "POST",
                "path": "/api/v1/alerts",
                "description": "Alertas de qualidade de dados",
                "parameters": "file (multipart/form-data)"
            },
            "visualizations": {
                "method": "POST",
                "path": "/api/v1/visualizations",
                "description": "Extrair visualizações SVG da análise",
                "parameters": "file (multipart/form-data)"
            },
            "variable_analysis": {
                "method": "POST",
                "path": "/api/v1/variable/{variable_name}",
                "description": "Análise detalhada de uma variável específica",
                "parameters": "variable_name (path), file (multipart/form-data)"
            },
            "supported_formats": {
                "method": "GET",
                "path": "/api/v1/supported-formats",
                "description": "Formatos de arquivo suportados"
            },
            "endpoints": {
                "method": "GET",
                "path": "/api/v1/endpoints", 
                "description": "Listar todos os endpoints (este endpoint)"
            },
            "r2_config": {
                "method": "GET",
                "path": "/api/v1/r2/config",
                "description": "Verificar configuração do Cloudflare R2"
            },
            "r2_presigned_upload": {
                "method": "POST",
                "path": "/api/v1/r2/presigned-upload",
                "description": "Gerar URL pré-assinada para upload no R2",
                "parameters": "filename (query), content_type (query, optional), folder (query, optional)"
            },
            "r2_presigned_download": {
                "method": "POST",
                "path": "/api/v1/r2/presigned-download",
                "description": "Gerar URL pré-assinada para download do R2",
                "parameters": "file_key (query)"
            },
            "r2_file_info": {
                "method": "GET",
                "path": "/api/v1/r2/file-info",
                "description": "Obter informações de um arquivo no R2",
                "parameters": "file_key (query)"
            },
            "r2_delete_file": {
                "method": "DELETE",
                "path": "/api/v1/r2/file",
                "description": "Deletar arquivo do R2",
                "parameters": "file_key (query)"
            },
            "r2_list_files": {
                "method": "GET",
                "path": "/api/v1/r2/files",
                "description": "Listar arquivos em uma pasta do R2",
                "parameters": "folder (query, optional), limit (query, optional)"
            },
            "r2_upload_and_process": {
                "method": "POST",
                "path": "/api/v1/r2/upload-and-process",
                "description": "Processar arquivo CSV já enviado para o R2",
                "parameters": "file_key (query)"
            }
        },
        "analysis_endpoints": {
            "start_analysis": {
                "method": "POST",
                "path": "/api/v1/analysis/start",
                "description": "Iniciar análise de arquivo já enviado para o R2 com suporte a opções de CSV",
                "parameters": "AnalysisStartRequest (body) - inclui file_key, analysis_type, options e csv_options"
            },
            "analysis_status": {
                "method": "GET", 
                "path": "/api/v1/analysis/status/{analysis_id}",
                "description": "Verificar status de análise em andamento",
                "parameters": "analysis_id (path)"
            },
            "analysis_results": {
                "method": "GET",
                "path": "/api/v1/analysis/results/{analysis_id}", 
                "description": "Obter resultados de análise concluída",
                "parameters": "analysis_id (path)"
            }
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }