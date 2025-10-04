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
class AnalysisStartRequest(BaseModel):
    """Request para iniciar análise"""
    file_key: str
    analysis_type: str = "basic_eda"
    options: Optional[Dict[str, Any]] = None

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

class ColumnStats(BaseModel):
    """Estatísticas de uma coluna"""
    name: str
    dtype: str
    count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    most_frequent: Optional[str] = None
    frequency: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None

class AnalysisSummary(BaseModel):
    """Resumo da análise"""
    completeness_score: float
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    recommendations: List[str]

class AnalysisResults(BaseModel):
    """Resultados completos da análise"""
    analysis_type: str
    dataset_info: DatasetInfo
    column_stats: List[ColumnStats]
    correlations: Dict[str, Any]
    summary: AnalysisSummary