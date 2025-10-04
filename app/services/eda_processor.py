"""
Processador EDA simplificado
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json


def convert_numpy_types(obj):
    """Converter tipos numpy para tipos Python serializáveis"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


class EDAProcessor:
    """Processador simplificado de análise EDA"""
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        analysis_type: str = "basic_eda",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Processar DataFrame e retornar análise EDA"""
        
        # Análise básica
        results = {
            "basic_info": self._get_basic_info(df),
            "statistical_summary": self._get_statistical_summary(df),
            "data_types": self._get_data_types(df),
            "missing_values": self._get_missing_values(df)
        }
        
        # Resumo para LLM
        summary = {
            "dataset_shape": df.shape,
            "total_records": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "missing_values_total": df.isnull().sum().sum(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # Visualizações básicas (placeholder)
        visualizations = {
            "available_charts": ["histogram", "boxplot", "correlation_heatmap"],
            "chart_count": 3
        }
        
        return {
            "results": convert_numpy_types(results),
            "summary": convert_numpy_types(summary),
            "visualizations": convert_numpy_types(visualizations)
        }
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Informações básicas do DataFrame"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "index_name": df.index.name,
            "memory_usage": df.memory_usage(deep=True).to_dict()
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Resumo estatístico"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return {"message": "Nenhuma coluna numérica encontrada"}
        
        # Converter para formato JSON serializável
        describe_result = numeric_df.describe()
        
        result = {}
        for col in describe_result.columns:
            result[col] = {}
            for stat in describe_result.index:
                value = describe_result.loc[stat, col]
                if pd.isna(value):
                    result[col][stat] = None
                else:
                    result[col][stat] = float(value)
        
        return result
    
    def _get_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Tipos de dados das colunas"""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def _get_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de valores faltantes"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            "counts": convert_numpy_types(missing_counts.to_dict()),
            "percentages": convert_numpy_types(missing_percentages.to_dict()),
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": list(missing_counts[missing_counts > 0].index)
        }