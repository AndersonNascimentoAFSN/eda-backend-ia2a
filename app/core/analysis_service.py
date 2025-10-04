"""
Serviço de análise de dados
"""
import io
import uuid
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.core.config import settings
from app.core.r2_service import r2_service
from app.models.analysis import (
    AnalysisRequest, AnalysisStatus, AnalysisResults, 
    DatasetInfo, ColumnStats
)

logger = logging.getLogger(__name__)


class DataAnalysisService:
    """Serviço para análise de dados de arquivos"""
    
    def __init__(self):
        # Em produção, isso seria um banco de dados ou cache
        self.analysis_cache: Dict[str, AnalysisResults] = {}
        self.status_cache: Dict[str, AnalysisStatus] = {}
    
    def start_analysis(self, request: AnalysisRequest) -> str:
        """
        Iniciar análise de arquivo
        
        Args:
            request: Dados da requisição de análise
            
        Returns:
            ID da análise criada
        """
        analysis_id = str(uuid.uuid4())
        
        # Criar status inicial
        status = AnalysisStatus(
            analysis_id=analysis_id,
            status="pending",
            progress=0.0,
            message="Análise iniciada",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_key=request.file_key,
            analysis_type=request.analysis_type
        )
        
        self.status_cache[analysis_id] = status
        
        # Em produção, isso seria uma tarefa assíncrona (Celery, etc.)
        try:
            self._process_analysis(analysis_id, request)
        except Exception as e:
            logger.error(f"Erro na análise {analysis_id}: {e}")
            self._update_status(analysis_id, "error", 0.0, f"Erro: {str(e)}")
        
        return analysis_id
    
    def get_analysis_status(self, analysis_id: str) -> Optional[AnalysisStatus]:
        """Obter status da análise"""
        return self.status_cache.get(analysis_id)
    
    def get_analysis_results(self, analysis_id: str) -> Optional[AnalysisResults]:
        """Obter resultados da análise"""
        return self.analysis_cache.get(analysis_id)
    
    def _process_analysis(self, analysis_id: str, request: AnalysisRequest):
        """Processar análise do arquivo"""
        try:
            # Atualizar status
            self._update_status(analysis_id, "processing", 10.0, "Baixando arquivo...")
            
            # Baixar arquivo do R2
            file_content = self._download_file_from_r2(request.file_key)
            
            self._update_status(analysis_id, "processing", 30.0, "Carregando dados...")
            
            # Carregar dados baseado no tipo de arquivo
            df = self._load_dataframe(file_content, request.file_key)
            
            self._update_status(analysis_id, "processing", 50.0, "Analisando dados...")
            
            # Realizar análise
            results = self._analyze_dataframe(analysis_id, df, request)
            
            self._update_status(analysis_id, "processing", 90.0, "Finalizando...")
            
            # Salvar resultados
            results.completed_at = datetime.now()
            self.analysis_cache[analysis_id] = results
            
            self._update_status(analysis_id, "completed", 100.0, "Análise concluída")
            
        except Exception as e:
            logger.error(f"Erro no processamento da análise {analysis_id}: {e}")
            self._update_status(analysis_id, "error", 0.0, f"Erro: {str(e)}")
            raise
    
    def _download_file_from_r2(self, file_key: str) -> bytes:
        """Baixar arquivo do R2"""
        try:
            response = r2_service.client.get_object(
                Bucket=settings.cloudflare_r2_bucket_name,
                Key=file_key
            )
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Erro ao baixar arquivo {file_key}: {e}")
            raise ValueError(f"Erro ao baixar arquivo: {e}")
    
    def _load_dataframe(self, file_content: bytes, file_key: str) -> pd.DataFrame:
        """Carregar DataFrame baseado no tipo de arquivo"""
        file_extension = Path(file_key).suffix.lower()
        
        try:
            if file_extension == '.csv':
                # Tentar diferentes encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Não foi possível decodificar o arquivo CSV")
                
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(io.BytesIO(file_content))
                
            elif file_extension == '.json':
                return pd.read_json(io.BytesIO(file_content))
                
            elif file_extension == '.parquet':
                return pd.read_parquet(io.BytesIO(file_content))
                
            else:
                raise ValueError(f"Tipo de arquivo não suportado: {file_extension}")
                
        except Exception as e:
            logger.error(f"Erro ao carregar DataFrame: {e}")
            raise ValueError(f"Erro ao processar arquivo: {e}")
    
    def _analyze_dataframe(self, analysis_id: str, df: pd.DataFrame, request: AnalysisRequest) -> AnalysisResults:
        """Realizar análise exploratória do DataFrame"""
        
        # Informações básicas do dataset
        dataset_info = DatasetInfo(
            filename=Path(request.file_key).name,
            file_size=df.memory_usage(deep=True).sum(),
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_usage=df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        )
        
        # Análise por coluna
        column_stats = []
        for col in df.columns:
            stats = self._analyze_column(df, col)
            column_stats.append(stats)
        
        # Correlações (apenas colunas numéricas)
        numeric_columns = df.select_dtypes(include=['number']).columns
        correlations = None
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            correlations = {
                col: corr_matrix[col].to_dict() 
                for col in numeric_columns
            }
        
        # Resumo executivo
        summary = self._generate_summary(df, dataset_info, column_stats)
        
        return AnalysisResults(
            analysis_id=analysis_id,
            status="completed",
            file_key=request.file_key,
            analysis_type=request.analysis_type,
            created_at=self.status_cache[analysis_id].created_at,
            dataset_info=dataset_info,
            column_stats=column_stats,
            correlations=correlations,
            summary=summary
        )
    
    def _analyze_column(self, df: pd.DataFrame, col: str) -> ColumnStats:
        """Analisar uma coluna específica"""
        series = df[col]
        
        stats = ColumnStats(
            name=col,
            dtype=str(series.dtype),
            count=len(series),
            non_null_count=series.count(),
            null_count=series.isnull().sum(),
            null_percentage=(series.isnull().sum() / len(series)) * 100,
            unique_count=series.nunique()
        )
        
        # Amostras de valores (primeiros 5 não-nulos)
        sample_values = series.dropna().head(5).tolist()
        stats.sample_values = sample_values
        
        # Estatísticas específicas por tipo
        if pd.api.types.is_numeric_dtype(series):
            # Colunas numéricas
            desc = series.describe()
            stats.mean = float(desc['mean']) if not pd.isna(desc['mean']) else None
            stats.std = float(desc['std']) if not pd.isna(desc['std']) else None
            stats.min = float(desc['min']) if not pd.isna(desc['min']) else None
            stats.max = float(desc['max']) if not pd.isna(desc['max']) else None
            stats.q25 = float(desc['25%']) if not pd.isna(desc['25%']) else None
            stats.q50 = float(desc['50%']) if not pd.isna(desc['50%']) else None
            stats.q75 = float(desc['75%']) if not pd.isna(desc['75%']) else None
            
        else:
            # Colunas categóricas/texto
            if stats.non_null_count > 0:
                value_counts = series.value_counts()
                if len(value_counts) > 0:
                    stats.most_frequent = str(value_counts.index[0])
                    stats.most_frequent_count = int(value_counts.iloc[0])
        
        return stats
    
    def _generate_summary(self, df: pd.DataFrame, dataset_info: DatasetInfo, column_stats: List[ColumnStats]) -> Dict[str, Any]:
        """Gerar resumo executivo da análise"""
        
        # Contadores por tipo de dados
        data_type_counts = {}
        for col_stat in column_stats:
            dtype_category = self._categorize_dtype(col_stat.dtype)
            data_type_counts[dtype_category] = data_type_counts.get(dtype_category, 0) + 1
        
        # Colunas com muitos valores nulos
        high_null_columns = [
            col_stat.name for col_stat in column_stats 
            if col_stat.null_percentage > 50
        ]
        
        # Colunas numéricas
        numeric_columns = [
            col_stat.name for col_stat in column_stats 
            if pd.api.types.is_numeric_dtype(df[col_stat.name])
        ]
        
        # Colunas categóricas com alta cardinalidade
        high_cardinality_columns = [
            col_stat.name for col_stat in column_stats 
            if col_stat.unique_count and col_stat.unique_count > (dataset_info.rows * 0.8)
        ]
        
        return {
            "dataset_shape": f"{dataset_info.rows:,} linhas × {dataset_info.columns} colunas",
            "memory_usage_mb": round(dataset_info.memory_usage, 2),
            "data_type_distribution": data_type_counts,
            "numeric_columns_count": len(numeric_columns),
            "high_null_columns": high_null_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "completeness_score": round(
                (sum(col_stat.non_null_count for col_stat in column_stats) / 
                 (dataset_info.rows * dataset_info.columns)) * 100, 1
            ),
            "recommendations": self._generate_recommendations(column_stats, dataset_info)
        }
    
    def _categorize_dtype(self, dtype: str) -> str:
        """Categorizar tipo de dados"""
        dtype_lower = dtype.lower()
        if any(t in dtype_lower for t in ['int', 'float', 'number']):
            return "numeric"
        elif any(t in dtype_lower for t in ['datetime', 'timestamp']):
            return "datetime"
        elif any(t in dtype_lower for t in ['bool']):
            return "boolean"
        else:
            return "categorical"
    
    def _generate_recommendations(self, column_stats: List[ColumnStats], dataset_info: DatasetInfo) -> List[str]:
        """Gerar recomendações baseadas na análise"""
        recommendations = []
        
        # Verificar colunas com muitos nulos
        high_null_cols = [col for col in column_stats if col.null_percentage > 30]
        if high_null_cols:
            recommendations.append(
                f"Considere tratar valores nulos em {len(high_null_cols)} coluna(s): "
                f"{', '.join([col.name for col in high_null_cols[:3]])}"
                + ("..." if len(high_null_cols) > 3 else "")
            )
        
        # Verificar colunas com alta cardinalidade
        high_card_cols = [
            col for col in column_stats 
            if col.unique_count and col.unique_count > (dataset_info.rows * 0.8)
        ]
        if high_card_cols:
            recommendations.append(
                f"Colunas com alta cardinalidade podem ser IDs únicos: "
                f"{', '.join([col.name for col in high_card_cols[:3]])}"
            )
        
        # Verificar dataset muito grande
        if dataset_info.memory_usage > 500:  # MB
            recommendations.append(
                f"Dataset grande ({dataset_info.memory_usage:.1f}MB). "
                "Considere usar amostragem para análises exploratórias."
            )
        
        return recommendations
    
    def _update_status(self, analysis_id: str, status: str, progress: float, message: str):
        """Atualizar status da análise"""
        if analysis_id in self.status_cache:
            current_status = self.status_cache[analysis_id]
            current_status.status = status
            current_status.progress = progress
            current_status.message = message
            current_status.updated_at = datetime.now()


# Instância global do serviço
analysis_service = DataAnalysisService()