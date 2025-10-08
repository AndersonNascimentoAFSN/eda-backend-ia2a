"""
Processador de análise exploratória de dados (EDA) usando ydata-profiling
"""
import json
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from ydata_profiling import ProfileReport


class EDAProcessor:
    """Processador de EDA para arquivos CSV"""
    
    def __init__(self):
        self.supported_extensions = ['.csv']
    
    def detect_separator(self, content: bytes) -> str:
        """
        Detecta automaticamente o separador do CSV
        
        Args:
            content: Conteúdo do arquivo em bytes
            
        Returns:
            Separador detectado
        """
        # Converter bytes para string para análise
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('latin-1')
            except UnicodeDecodeError:
                text_content = content.decode('cp1252', errors='ignore')
        
        # Lista de separadores possíveis em ordem de prioridade
        separators = [';', ',', '\t', '|']
        
        # Obter as primeiras linhas para análise
        lines = text_content.split('\n')[:5]
        if not lines:
            return ','  # Default
        
        best_separator = ','
        max_columns = 1
        
        for sep in separators:
            try:
                # Tentar ler com este separador
                df_test = pd.read_csv(StringIO(text_content), sep=sep, nrows=3)
                num_columns = len(df_test.columns)
                
                # Se conseguiu mais de 1 coluna e não tem colunas com nomes suspeitos
                if num_columns > max_columns:
                    # Verificar se não há nomes de coluna muito longos (indicando separador errado)
                    max_col_name_length = max(len(str(col)) for col in df_test.columns)
                    if max_col_name_length < 100:  # Limite razoável
                        max_columns = num_columns
                        best_separator = sep
                        
            except Exception:
                continue
        
        print(f"🔍 Separador detectado: '{best_separator}' (resultou em {max_columns} colunas)")
        return best_separator
    
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
            # Detectar separador automaticamente
            separator = self.detect_separator(content)
            
            # Tentar ler as primeiras linhas para validar
            df_sample = pd.read_csv(BytesIO(content), sep=separator, nrows=5)
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
            # Detectar separador automaticamente
            separator = self.detect_separator(content)
            print(f"📊 Processando CSV com separador: '{separator}'")
            
            # Carregar dados com separador correto
            df = pd.read_csv(BytesIO(content), sep=separator)
            
            print(f"✅ CSV carregado: {len(df)} linhas, {len(df.columns)} colunas")
            print(f"📋 Colunas: {df.columns.tolist()}")
            
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
            # Detectar separador automaticamente
            separator = self.detect_separator(content)
            print(f"📊 Analisando CSV com separador: '{separator}'")
            
            # Carregar dados com separador correto
            df = pd.read_csv(BytesIO(content), sep=separator)
            
            print(f"✅ CSV carregado: {len(df)} linhas, {len(df.columns)} colunas")
            print(f"📋 Colunas: {df.columns.tolist()}")
            
            return {
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "has_null_values": df.isnull().any().any(),
                "null_counts": df.isnull().sum().to_dict(),
                "detected_separator": separator
            }
        except Exception as e:
            raise RuntimeError(f"Erro ao obter informações básicas: {str(e)}")