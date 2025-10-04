"""
Serviço de análise de dados EDA (Exploratory Data Analysis)
"""
import io
import json
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging

from app.core.r2_service import r2_service
from app.services.visualization_service import visualization_service
from app.services.advanced_stats_service import advanced_stats_service
from app.services.temporal_analysis_service import temporal_analysis_service
from app.services.statistical_tests_service import statistical_tests_service

logger = logging.getLogger(__name__)

class DataAnalysisStatus:
    """Estados possíveis da análise"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    ERROR = "error"

class DataAnalyzer:
    """Serviço principal de análise de dados"""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def start_analysis(
        self, 
        file_key: str, 
        analysis_type: str = "basic_eda",
        options: Optional[Dict] = None
    ) -> str:
        """
        Iniciar análise assíncrona de um arquivo no R2
        
        Args:
            file_key: Chave do arquivo no R2
            analysis_type: Tipo de análise (basic_eda, advanced_stats, etc.)
            options: Opções adicionais da análise
            
        Returns:
            ID único da análise
        """
        analysis_id = str(uuid.uuid4())
        
        # Inicializar status da análise
        self.analysis_cache[analysis_id] = {
            "id": analysis_id,
            "file_key": file_key,
            "analysis_type": analysis_type,
            "status": DataAnalysisStatus.PENDING,
            "progress": 0.0,
            "message": "Análise iniciada",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "results": None,
            "error": None,
            "options": options or {}
        }
        
        # Executar análise em background
        asyncio.create_task(self._run_analysis(analysis_id))
        
        logger.info(f"Análise {analysis_id} iniciada para arquivo {file_key}")
        return analysis_id
    
    async def _run_analysis(self, analysis_id: str):
        """Executar análise em background"""
        try:
            analysis = self.analysis_cache[analysis_id]
            
            # Atualizar status
            analysis["status"] = DataAnalysisStatus.PROCESSING
            analysis["progress"] = 10.0
            analysis["message"] = "Baixando arquivo do R2"
            
            # 1. Baixar arquivo do R2
            file_content = await self._download_file_from_r2(analysis["file_key"])
            
            analysis["progress"] = 30.0
            analysis["message"] = "Carregando dados"
            
            # 2. Carregar dados
            df = await self._load_dataframe(file_content, analysis["file_key"])
            
            analysis["progress"] = 50.0
            analysis["message"] = "Executando análise"
            
            # 3. Executar análise baseada no tipo
            if analysis["analysis_type"] == "basic_eda":
                results = await self._basic_eda_analysis(df, analysis["file_key"])
            elif analysis["analysis_type"] == "advanced_stats":
                results = await self._advanced_stats_analysis(df)
            elif analysis["analysis_type"] == "data_quality":
                results = await self._data_quality_analysis(df)
            else:
                raise ValueError(f"Tipo de análise não suportado: {analysis['analysis_type']}")
            
            # 4. Finalizar
            analysis["status"] = DataAnalysisStatus.COMPLETED
            analysis["progress"] = 100.0
            analysis["message"] = "Análise concluída"
            analysis["completed_at"] = datetime.now().isoformat()
            analysis["results"] = results
            
            logger.info(f"Análise {analysis_id} concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na análise {analysis_id}: {e}")
            analysis = self.analysis_cache[analysis_id]
            analysis["status"] = DataAnalysisStatus.ERROR
            analysis["message"] = f"Erro: {str(e)}"
            analysis["error"] = str(e)
            analysis["completed_at"] = datetime.now().isoformat()
    
    async def _download_file_from_r2(self, file_key: str) -> bytes:
        """Baixar arquivo do R2"""
        if not r2_service.is_configured():
            raise ValueError("R2 não está configurado")
        
        try:
            # Gerar URL de download
            download_data = r2_service.generate_presigned_download_url(file_key)
            
            # Fazer download do arquivo (simulação - em produção usar aiohttp)
            import requests
            response = requests.get(download_data["download_url"])
            
            if response.status_code != 200:
                raise ValueError(f"Erro ao baixar arquivo: {response.status_code}")
            
            return response.content
            
        except Exception as e:
            raise ValueError(f"Erro ao baixar arquivo do R2: {e}")
    
    async def _load_dataframe(self, file_content: bytes, file_key: str) -> pd.DataFrame:
        """Carregar arquivo em DataFrame"""
        try:
            # Detectar tipo de arquivo pela extensão
            file_extension = Path(file_key).suffix.lower()
            
            if file_extension == '.csv':
                # Tentar diferentes encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Não foi possível decodificar o arquivo CSV")
                    
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(file_content))
                
            elif file_extension == '.json':
                json_data = json.loads(file_content.decode('utf-8'))
                df = pd.json_normalize(json_data)
                
            else:
                raise ValueError(f"Tipo de arquivo não suportado: {file_extension}")
            
            if df.empty:
                raise ValueError("Arquivo está vazio")
            
            logger.info(f"Arquivo carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
            
        except Exception as e:
            raise ValueError(f"Erro ao carregar dados: {e}")
    
    async def _basic_eda_analysis(self, df: pd.DataFrame, file_key: str) -> Dict[str, Any]:
        """Análise exploratória básica"""
        
        # Informações básicas do dataset
        dataset_info = {
            "filename": Path(file_key).name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            "dtypes": df.dtypes.value_counts().to_dict()
        }
        
        # Análise por coluna
        column_stats = []
        for col in df.columns:
            col_data = df[col]
            
            stats = {
                "name": col,
                "dtype": str(col_data.dtype),
                "count": len(col_data),
                "non_null_count": col_data.count(),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
                "unique_count": col_data.nunique(),
                "most_frequent": None,
                "frequency": None
            }
            
            # Estatísticas numéricas
            if pd.api.types.is_numeric_dtype(col_data):
                non_null_data = col_data.dropna()
                if not non_null_data.empty:
                    stats.update({
                        "mean": float(non_null_data.mean()),
                        "median": float(non_null_data.median()),
                        "std": float(non_null_data.std()),
                        "variance": float(non_null_data.var()),
                        "min": float(non_null_data.min()),
                        "max": float(non_null_data.max()),
                        "q25": float(non_null_data.quantile(0.25)),
                        "q75": float(non_null_data.quantile(0.75)),
                        "skewness": float(non_null_data.skew()),
                        "kurtosis": float(non_null_data.kurtosis()),
                        "range": float(non_null_data.max() - non_null_data.min()),
                        "iqr": float(non_null_data.quantile(0.75) - non_null_data.quantile(0.25))
                    })
                    
                    # Detecção de outliers usando IQR
                    Q1 = non_null_data.quantile(0.25)
                    Q3 = non_null_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = non_null_data[(non_null_data < Q1 - 1.5 * IQR) | 
                                           (non_null_data > Q3 + 1.5 * IQR)]
                    stats.update({
                        "outlier_count": len(outliers),
                        "outlier_percentage": (len(outliers) / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0,
                        "outlier_bounds": {
                            "lower": float(Q1 - 1.5 * IQR),
                            "upper": float(Q3 + 1.5 * IQR)
                        }
                    })
            else:
                # Para colunas categóricas
                if not col_data.empty and col_data.count() > 0:
                    value_counts = col_data.value_counts()
                    stats.update({
                        "most_frequent": str(value_counts.index[0]),
                        "frequency": int(value_counts.iloc[0]),
                        "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                        "least_frequency": int(value_counts.iloc[-1]) if len(value_counts) > 0 else None,
                        "cardinality": len(value_counts),
                        "top_values": value_counts.head(5).to_dict()
                    })
                    
                    # Verificar se pode ser coluna temporal
                    if col_data.dtype == 'object':
                        try:
                            # Tentar converter uma amostra para datetime
                            sample = col_data.dropna().head(10)
                            pd.to_datetime(sample)
                            stats["potential_datetime"] = True
                        except:
                            stats["potential_datetime"] = False
            
            column_stats.append(stats)
        
        # Correlações (apenas para colunas numéricas)
        numeric_columns = df.select_dtypes(include=[np.number])
        correlations = {}
        if len(numeric_columns.columns) > 1:
            corr_matrix = numeric_columns.corr()
            correlations = corr_matrix.to_dict()
            
            # Adicionar correlações mais fortes
            strong_correlations = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Evitar duplicatas
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.5:  # Correlação significativa
                            strong_correlations.append({
                                "variable1": col1,
                                "variable2": col2,
                                "correlation": float(corr_val),
                                "strength": "strong" if abs(corr_val) > 0.7 else "moderate"
                            })
            
            correlations["strong_correlations"] = strong_correlations
        
        # Resumo e recomendações
        completeness_score = ((df.count().sum() / (len(df) * len(df.columns))) * 100).round(1)
        
        recommendations = []
        
        # Verificar colunas com muitos valores faltantes
        high_null_cols = [col for col in df.columns if (df[col].isnull().sum() / len(df)) > 0.5]
        if high_null_cols:
            recommendations.append(f"Colunas com >50% valores faltantes: {', '.join(high_null_cols[:3])}")
        
        # Verificar colunas categóricas com alta cardinalidade
        high_cardinality = [col for col in df.select_dtypes(include=['object']).columns 
                          if df[col].nunique() > len(df) * 0.8]
        if high_cardinality:
            recommendations.append(f"Possíveis IDs únicos: {', '.join(high_cardinality[:3])}")
        
        # Verificar outliers em colunas numéricas
        outlier_cols = []
        for col in numeric_columns.columns:
            Q1 = numeric_columns[col].quantile(0.25)
            Q3 = numeric_columns[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = numeric_columns[col][(numeric_columns[col] < Q1 - 1.5 * IQR) | 
                                          (numeric_columns[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                outlier_cols.append(col)
        
        if outlier_cols:
            recommendations.append(f"Possíveis outliers em: {', '.join(outlier_cols[:3])}")
        
        # Recomendações de correlação
        if correlations.get("strong_correlations"):
            strong_corrs = correlations["strong_correlations"]
            if len(strong_corrs) > 0:
                recommendations.append(f"Correlações fortes detectadas entre {len(strong_corrs)} pares de variáveis")
        
        # Recomendações de qualidade geral
        if completeness_score >= 95:
            recommendations.append("Dataset com alta qualidade de dados")
        elif completeness_score >= 80:
            recommendations.append("Considerar limpeza de dados antes da análise")
        else:
            recommendations.append("Dataset requer limpeza significativa")
        
        # Detectar possíveis colunas temporais
        potential_datetime_cols = [col["name"] for col in column_stats if col.get("potential_datetime", False)]
        if potential_datetime_cols:
            recommendations.append(f"Possíveis colunas de data/hora: {', '.join(potential_datetime_cols[:3])}")
        
        # Verificar variabilidade baixa
        low_variance_cols = []
        for col_stat in column_stats:
            if col_stat.get("std") is not None and col_stat["std"] < 0.01:
                low_variance_cols.append(col_stat["name"])
        
        if low_variance_cols:
            recommendations.append(f"Colunas com baixa variabilidade: {', '.join(low_variance_cols[:3])}")
        
        # Contagem de tipos de colunas
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        categorical_cols = df.select_dtypes(include=['object', 'category'])
        datetime_cols = df.select_dtypes(include=['datetime64'])
        
        logger.info(f"Tipos de colunas detectados - Numeric: {len(numeric_cols.columns)}, Categorical: {len(categorical_cols.columns)}, Datetime: {len(datetime_cols.columns)}")
        logger.info(f"Colunas numéricas: {list(numeric_cols.columns)}")
        logger.info(f"Colunas categóricas: {list(categorical_cols.columns)}")
        
        return {
            "analysis_type": "basic_eda",
            "dataset_info": dataset_info,
            "column_stats": column_stats,
            "correlations": correlations,
            "data_quality": {
                "completeness_score": float(completeness_score),
                "total_missing_values": int(df.isnull().sum().sum()),
                "columns_with_missing": len([col for col in df.columns if df[col].isnull().sum() > 0]),
                "duplicate_rows": int(df.duplicated().sum()),
                "potential_datetime_columns": potential_datetime_cols,
                "high_cardinality_columns": high_cardinality,
                "low_variance_columns": low_variance_cols
            },
            "summary": {
                "completeness_score": float(completeness_score),
                "numeric_columns": len(numeric_cols.columns),
                "categorical_columns": len(categorical_cols.columns),
                "datetime_columns": len(datetime_cols.columns),
                "total_outliers": sum([col.get("outlier_count", 0) for col in column_stats]),
                "strong_correlations_count": len(correlations.get("strong_correlations", [])),
                "recommendations": recommendations
            }
        }
    
    async def _advanced_stats_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise estatística avançada"""
        # Implementar análises mais avançadas
        return {"message": "Análise avançada em desenvolvimento"}
    
    async def _data_quality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de qualidade de dados"""
        # Implementar análise de qualidade
        return {"message": "Análise de qualidade em desenvolvimento"}
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict]:
        """Obter status da análise"""
        return self.analysis_cache.get(analysis_id)
    
    def get_analysis_results(self, analysis_id: str) -> Optional[Dict]:
        """Obter resultados da análise"""
        analysis = self.analysis_cache.get(analysis_id)
        if analysis and analysis["status"] == DataAnalysisStatus.COMPLETED:
            return analysis["results"]
        return None
    
    def cleanup_analysis(self, analysis_id: str) -> bool:
        """Limpar análise do cache"""
        if analysis_id in self.analysis_cache:
            del self.analysis_cache[analysis_id]
            return True
        return False

# Instância global do analisador
data_analyzer = DataAnalyzer()