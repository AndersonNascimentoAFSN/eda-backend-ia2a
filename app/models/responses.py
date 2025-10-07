"""
Modelos de resposta para a API EDA
"""
from typing import Any, Dict, Optional, List
from pydantic import BaseModel


class EDAResponse(BaseModel):
    """Resposta do processamento EDA"""
    success: bool
    message: str
    filename: str
    eda_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Resposta de erro"""
    success: bool = False
    message: str
    error: str


class HealthResponse(BaseModel):
    """Resposta de health check"""
    status: str = "healthy"
    message: str = "EDA Backend API is running"


class PresignedUploadResponse(BaseModel):
    """Resposta com URL pré-assinada para upload"""
    success: bool
    upload_url: str
    method: str = "PUT"
    file_key: str
    content_type: str
    expires_in: int
    max_file_size_mb: int
    expires_at: str
    headers: Dict[str, str]
    instructions: Dict[str, str] = {
        "method": "PUT",
        "description": "Use a PUT request to upload the file directly to the URL",
        "example": "curl -X PUT 'upload_url' -H 'Content-Type: content_type' --data-binary @file"
    }


class PresignedDownloadResponse(BaseModel):
    """Resposta com URL pré-assinada para download"""
    success: bool
    download_url: str
    file_key: str
    expires_in: int
    expires_at: str


class FileInfoResponse(BaseModel):
    """Resposta com informações do arquivo"""
    success: bool
    file_key: str
    size: int
    content_type: str
    last_modified: str
    etag: str
    metadata: Dict[str, str]
    original_filename: str


class FileListResponse(BaseModel):
    """Resposta com lista de arquivos"""
    success: bool
    files: List[Dict[str, Any]]
    count: int
    folder: str
    truncated: bool


class R2ConfigResponse(BaseModel):
    """Resposta da configuração do R2"""
    configured: bool
    bucket_name: Optional[str] = None
    message: str

# Modelos para análise de dados
class CSVOptions(BaseModel):
    """Opções específicas para leitura de arquivos CSV"""
    sep: Optional[str] = None  # Separador (vírgula, ponto-e-vírgula, etc.)
    encoding: Optional[str] = None  # utf-8, latin-1, cp1252, etc.
    decimal: Optional[str] = None  # Separador decimal (. ou ,)
    thousands: Optional[str] = None  # Separador de milhares
    parse_dates: Optional[List[str]] = None  # Colunas a serem interpretadas como datas
    date_format: Optional[str] = None  # Formato das datas
    dtype: Optional[Dict[str, str]] = None  # Tipos específicos para colunas
    na_values: Optional[List[str]] = None  # Valores a serem tratados como NaN
    quotechar: Optional[str] = None  # Caractere de aspas
    quoting: Optional[int] = None  # Comportamento de aspas (0=minimal, 1=all, 2=non-numeric, 3=none)
    skiprows: Optional[int] = None  # Número de linhas para pular
    nrows: Optional[int] = None  # Número máximo de linhas para ler
    header: Optional[int] = None  # Linha do cabeçalho (padrão=0)

class AnalysisStartRequest(BaseModel):
    """Request para iniciar análise"""
    file_key: str
    analysis_type: str = "basic_eda"
    options: Optional[Dict[str, Any]] = None
    csv_options: Optional[CSVOptions] = None  # Opções específicas para CSV

class AnalysisStartResponse(BaseModel):
    """Resposta ao iniciar análise"""
    analysis_id: str
    status: str
    message: str
    estimated_duration_minutes: Optional[int] = None

class AnalysisStatusResponse(BaseModel):
    """Status da análise"""
    analysis_id: str
    status: str
    progress: float
    message: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

class DatasetInfo(BaseModel):
    """Informações básicas do dataset"""
    filename: str
    rows: int
    columns: int
    memory_usage: float
    dtypes: Dict[str, int]
    file_size: Optional[int] = None
    column_names: Optional[List[str]] = None
    data_types: Optional[Dict[str, str]] = None

class ColumnStats(BaseModel):
    """Estatísticas de uma coluna (flexível para diferentes análises)"""
    name: str
    dtype: str
    
    # Campos básicos (opcionais para flexibilidade)
    count: Optional[int] = None
    non_null_count: Optional[int] = None
    null_count: Optional[int] = None
    null_percentage: Optional[float] = None
    unique_count: Optional[int] = None
    
    # Campos opcionais que podem existir
    most_frequent: Optional[str] = None
    frequency: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    
    # Permitir campos adicionais para advanced_stats
    class Config:
        extra = "allow"

class AnalysisSummary(BaseModel):
    """Resumo flexível da análise para diferentes tipos"""
    # Campos comuns (opcionais para flexibilidade)
    completeness_score: Optional[float] = None
    numeric_columns: Optional[int] = None
    categorical_columns: Optional[int] = None
    datetime_columns: Optional[int] = None
    recommendations: Optional[List[str]] = None
    
    # Campos específicos para advanced_stats (opcionais)
    analysis_type: Optional[str] = None
    dataset_health_score: Optional[float] = None
    key_findings: Optional[List[str]] = None
    data_distribution_summary: Optional[Dict[str, Any]] = None
    relationship_strength: Optional[Dict[str, Any]] = None
    anomaly_summary: Optional[Dict[str, Any]] = None
    next_steps: Optional[List[str]] = None
    
    # Permitir campos adicionais
    class Config:
        extra = "allow"

class CorrelationInfo(BaseModel):
    """Informação de correlação"""
    variable1: str
    variable2: str
    correlation: float
    strength: str

class CorrelationResults(BaseModel):
    """Resultados de correlação flexível para different analysis types"""
    correlations: Dict[str, Any]  # Pode ser Dict[str, float] ou Dict[str, Dict[str, float]]
    strong_correlations: List[Dict[str, Any]]  # Flexível para diferentes formatos

class AnalysisResults(BaseModel):
    """Resultados completos da análise"""
    analysis_type: str
    dataset_info: DatasetInfo
    column_stats: List[ColumnStats]
    correlations: CorrelationResults
    data_quality: Dict[str, Any]
    summary: AnalysisSummary

class AnalysisResultResponse(BaseModel):
    """Resposta completa com resultados da análise"""
    analysis_id: str
    status: str
    file_key: str
    created_at: str
    completed_at: Optional[str] = None
    results: AnalysisResults