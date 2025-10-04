"""
Modelos para análise de dados
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request para iniciar análise de arquivo"""
    file_key: str = Field(..., description="Chave do arquivo no R2")
    analysis_type: str = Field(default="basic_eda", description="Tipo de análise a ser realizada")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Opções específicas da análise")


class AnalysisStatus(BaseModel):
    """Status da análise"""
    analysis_id: str
    status: str  # pending, processing, completed, error
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    file_key: str
    analysis_type: str


class DatasetInfo(BaseModel):
    """Informações básicas do dataset"""
    filename: str
    file_size: int
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    memory_usage: float  # MB


class ColumnStats(BaseModel):
    """Estatísticas de uma coluna"""
    name: str
    dtype: str
    count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: Optional[int] = None
    
    # Para colunas numéricas
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None  # mediana
    q75: Optional[float] = None
    
    # Para colunas categóricas
    most_frequent: Optional[str] = None
    most_frequent_count: Optional[int] = None
    
    # Amostras de valores
    sample_values: List[Any] = Field(default_factory=list)


class AnalysisResults(BaseModel):
    """Resultados completos da análise"""
    analysis_id: str
    status: str
    file_key: str
    analysis_type: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    # Informações do dataset
    dataset_info: Optional[DatasetInfo] = None
    
    # Estatísticas por coluna
    column_stats: List[ColumnStats] = Field(default_factory=list)
    
    # Correlações (apenas para colunas numéricas)
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    
    # Resumo executivo
    summary: Optional[Dict[str, Any]] = None
    
    # Mensagens de erro (se houver)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)