"""
Processador de análise exploratória de dados (EDA) usando ydata-profiling
"""
import json
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from ydata_profiling import ProfileReport


class EDAProcessor:
    """Processador de EDA para arquivos CSV"""
    
    def __init__(self):
        self.supported_extensions = ['.csv']
    
    def validate_file(self, filename: str, content: bytes) -> Tuple[bool, str]:
        """
        Valida se o arquivo é um CSV válido
        
        Args:
            filename: Nome do arquivo
            content: Conteúdo do arquivo em bytes
            
        Returns:
            Tuple com (is_valid, error_message)
        """
        # Verificar extensão
        file_path = Path(filename)
        if file_path.suffix.lower() not in self.supported_extensions:
            return False, f"Tipo de arquivo não suportado. Use: {', '.join(self.supported_extensions)}"
        
        # Verificar se o conteúdo pode ser lido como CSV
        try:
            # Tentar ler as primeiras linhas para validar
            df_sample = pd.read_csv(BytesIO(content), nrows=5)
            if df_sample.empty:
                return False, "Arquivo CSV está vazio"
            return True, ""
        except Exception as e:
            return False, f"Erro ao ler arquivo CSV: {str(e)}"
    
    def process_csv(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Processa o arquivo CSV e gera relatório EDA
        
        Args:
            filename: Nome do arquivo
            content: Conteúdo do arquivo em bytes
            
        Returns:
            Dicionário com dados da análise EDA
        """
        # Validar arquivo
        is_valid, error_msg = self.validate_file(filename, content)
        if not is_valid:
            raise ValueError(error_msg)
        
        try:
            # Carregar dados
            df = pd.read_csv(BytesIO(content))
            
            # Gerar relatório com configurações otimizadas
            profile = ProfileReport(
                df,
                title=f"EDA Report - {filename}",
                explorative=True,
                minimal=False,
                samples={
                    "head": 5,
                    "tail": 5
                },
                correlations={
                    "auto": {"calculate": True},
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": True},
                    "cramers": {"calculate": True},
                },
                missing_diagrams={
                    "matrix": True,
                    "bar": True,
                    "heatmap": True,
                    "dendrogram": True,
                },
                interactions={
                    "continuous": True,
                    "targets": []
                },
                progress_bar=False
            )
            
            # Converter para JSON
            eda_json = profile.to_json()
            eda_data = json.loads(eda_json)
            
            return {
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "eda_report": eda_data
            }
            
        except Exception as e:
            raise RuntimeError(f"Erro ao processar arquivo: {str(e)}")
    
    def get_basic_info(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Obtém informações básicas do CSV sem gerar relatório completo
        
        Args:
            filename: Nome do arquivo
            content: Conteúdo do arquivo em bytes
            
        Returns:
            Dicionário com informações básicas
        """
        try:
            df = pd.read_csv(BytesIO(content))
            
            return {
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "has_null_values": df.isnull().any().any(),
                "null_counts": df.isnull().sum().to_dict()
            }
        except Exception as e:
            raise RuntimeError(f"Erro ao obter informações básicas: {str(e)}")