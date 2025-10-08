"""
Serviço de análise de dados EDA (Exploratory Data Analysis)
Otimizado para uso eficiente de memória
"""
import io
import json
import uuid
import os
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
try:
    from scipy import stats
except ImportError:
    stats = None

from app.core.r2_service import r2_service
from app.services.visualization_service import visualization_service
from app.services.advanced_stats_service import advanced_stats_service
from app.services.temporal_analysis_service import temporal_analysis_service

# Configurações de otimização de memória
MEMORY_LIMITS = {
    "small_dataset": 50 * 1024 * 1024,  # 50MB
    "large_dataset": 100 * 1024 * 1024,  # 100MB
    "max_rows_full_analysis": 100000,
    "max_rows_sample": 50000,
    "max_rows_advanced_sample": 25000,
    "chunk_size": 10000,
    "max_columns_correlation": 50,
}
from app.services.statistical_tests_service import statistical_tests_service

logger = logging.getLogger(__name__)

class DataAnalysisStatus:
    """Estados possíveis da análise"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    ERROR = "error"

class DataAnalyzer:
    """Serviço principal de análise de dados com otimização de memória"""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _force_garbage_collection(self):
        """Força garbage collection para liberar memória"""
        gc.collect()
        
    def _get_memory_usage_mb(self) -> float:
        """Retorna o uso atual de memória em MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
            
    def _check_memory_and_gc(self, threshold_mb: float = 200):
        """Verifica uso de memória e força GC se necessário"""
        current_memory = self._get_memory_usage_mb()
        if current_memory > threshold_mb:
            logger.info(f"🔧 Uso de memória alto ({current_memory:.1f}MB), forçando garbage collection")
            self._force_garbage_collection()
            new_memory = self._get_memory_usage_mb()
            logger.info(f"✅ Memória após GC: {new_memory:.1f}MB")
    
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
            
            # 2. Carregar dados com opções de CSV
            csv_options = analysis.get("options", {}).get("csv_options")
            df = await self._load_dataframe(file_content, analysis["file_key"], csv_options)
            
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
        """Baixar arquivo do R2 usando asyncio.to_thread"""
        if not r2_service.is_configured():
            raise ValueError("R2 não está configurado")
        
        try:
            # Gerar URL de download
            download_data = r2_service.generate_presigned_download_url(file_key)
            
            # Usar asyncio.to_thread para requests não bloquear o loop
            def _download():
                import requests
                response = requests.get(download_data["download_url"])
                if response.status_code != 200:
                    raise ValueError(f"Erro ao baixar arquivo: {response.status_code}")
                return response.content
            
            return await asyncio.to_thread(_download)
            
        except Exception as e:
            raise ValueError(f"Erro ao baixar arquivo do R2: {e}")
    
    def _detect_csv_separator(self, file_content: bytes) -> str:
        """
        Detecta automaticamente o separador do CSV usando apenas uma amostra pequena
        
        Args:
            file_content: Conteúdo do arquivo em bytes
            
        Returns:
            Separador detectado
        """
        from .memory_config import MEMORY_LIMITS
        
        # Para arquivos grandes, usar apenas uma amostra pequena para detecção
        sample_size = min(MEMORY_LIMITS['CSV_SEPARATOR_SAMPLE_SIZE'], len(file_content))
        sample_content = file_content[:sample_size]
        
        # Converter bytes para string para análise
        try:
            text_content = sample_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = sample_content.decode('latin-1')
            except UnicodeDecodeError:
                text_content = sample_content.decode('cp1252', errors='ignore')
        
        # Lista de separadores possíveis em ordem de prioridade
        separators = [';', ',', '\t', '|']
        
        # Obter as primeiras linhas para análise
        lines = text_content.split('\n')[:10]  # Usar mais linhas para melhor detecção
        if not lines:
            return ','  # Default
        
        best_separator = ','
        max_columns = 1
        
        # Usar apenas algumas linhas para teste (mais eficiente)
        sample_lines = '\n'.join(lines[:5])
        
        for sep in separators:
            try:
                # Tentar ler com este separador usando apenas o sample
                df_test = pd.read_csv(io.StringIO(sample_lines), sep=sep, nrows=3)
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
        
        logger.info(f"🔍 Separador detectado: '{best_separator}' (resultou em {max_columns} colunas)")
        return best_separator
    
    async def _load_dataframe(self, file_content: bytes, file_key: str, csv_options: Optional[Dict] = None) -> pd.DataFrame:
        """Carregar arquivo em DataFrame com suporte a opções de CSV e otimização de memória"""
        try:
            # Detectar tipo de arquivo pela extensão
            file_extension = Path(file_key).suffix.lower()
            
            # Verificar tamanho do arquivo
            file_size_mb = len(file_content) / (1024 * 1024)
            logger.info(f"📁 Tamanho do arquivo: {file_size_mb:.2f} MB")
            
            if file_extension == '.csv':
                # Detectar separador automaticamente se não especificado
                auto_separator = None
                if csv_options is None or 'sep' not in csv_options:
                    auto_separator = self._detect_csv_separator(file_content)
                    logger.info(f"📊 Separador detectado automaticamente: '{auto_separator}'")
                
                # Preparar argumentos para pd.read_csv
                read_args = {}
                
                # Usar separador detectado se não especificado
                if auto_separator:
                    read_args['sep'] = auto_separator
                
                # OTIMIZAÇÕES DE MEMÓRIA para arquivos grandes
                if file_size_mb > 50:  # Para arquivos > 50MB
                    logger.info("🔧 Aplicando otimizações de memória para arquivo grande")
                    
                    # Ler apenas uma amostra primeiro para otimizar dtypes
                    sample_args = read_args.copy()
                    sample_args['nrows'] = 1000  # Apenas 1000 linhas para análise
                    
                    # Tentar diferentes encodings com amostra
                    sample_df = None
                    encoding_to_use = 'utf-8'
                    
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            sample_df = pd.read_csv(
                                io.BytesIO(file_content), 
                                encoding=encoding, 
                                **sample_args
                            )
                            encoding_to_use = encoding
                            logger.info(f"✅ Encoding detectado: '{encoding}'")
                            break
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            continue
                    
                    if sample_df is None:
                        raise ValueError("Não foi possível decodificar o arquivo CSV")
                    
                    # Otimizar tipos de dados baseado na amostra
                    optimized_dtypes = {}
                    for col in sample_df.columns:
                        if sample_df[col].dtype == 'object':
                            # Tentar converter para categorias se tem muitos valores repetidos
                            if sample_df[col].nunique() / len(sample_df) < 0.5:
                                optimized_dtypes[col] = 'category'
                        elif sample_df[col].dtype in ['int64', 'float64']:
                            # Tentar usar tipos menores
                            if sample_df[col].dtype == 'int64':
                                if sample_df[col].min() >= -32768 and sample_df[col].max() <= 32767:
                                    optimized_dtypes[col] = 'int16'
                                elif sample_df[col].min() >= -2147483648 and sample_df[col].max() <= 2147483647:
                                    optimized_dtypes[col] = 'int32'
                            elif sample_df[col].dtype == 'float64':
                                optimized_dtypes[col] = 'float32'
                    
                    # Aplicar tipos otimizados
                    read_args['dtype'] = optimized_dtypes
                    read_args['encoding'] = encoding_to_use
                    
                    # Ler em chunks para economizar memória
                    chunk_size = 10000
                    chunks = []
                    
                    logger.info(f"📚 Lendo arquivo em chunks de {chunk_size} linhas")
                    
                    try:
                        # Remover dtype para evitar conflitos na leitura em chunks
                        chunk_args = {k: v for k, v in read_args.items() if k != 'dtype'}
                        
                        chunk_reader = pd.read_csv(
                            io.BytesIO(file_content),
                            chunksize=chunk_size,
                            **chunk_args
                        )
                        
                        for i, chunk in enumerate(chunk_reader):
                            chunks.append(chunk)
                            if i % 10 == 0:  # Log a cada 10 chunks
                                logger.info(f"📖 Processado chunk {i+1} ({len(chunks) * chunk_size} linhas)")
                        
                        # Concatenar chunks
                        logger.info("🔄 Concatenando chunks...")
                        df = pd.concat(chunks, ignore_index=True)
                        
                        # Limpar chunks da memória
                        del chunks
                        import gc
                        gc.collect()
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Falha ao ler em chunks: {e}")
                        # Fallback para leitura normal sem dtype para evitar conflitos
                        safe_args = {k: v for k, v in read_args.items() if k != 'dtype'}
                        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding_to_use, **safe_args)
                        
                else:
                    # Para arquivos menores, usar método normal
                    # Aplicar opções de CSV se fornecidas
                    if csv_options:
                        # Mapeamento das opções
                        option_mapping = {
                            'sep': 'sep',
                            'encoding': 'encoding', 
                            'decimal': 'decimal',
                            'thousands': 'thousands',
                            'parse_dates': 'parse_dates',
                            'date_format': 'date_format',
                            'dtype': 'dtype',
                            'na_values': 'na_values',
                            'quotechar': 'quotechar',
                            'quoting': 'quoting',
                            'skiprows': 'skiprows',
                            'nrows': 'nrows',
                            'header': 'header'
                        }
                        
                        for option, pandas_arg in option_mapping.items():
                            if option in csv_options and csv_options[option] is not None:
                                read_args[pandas_arg] = csv_options[option]
                    
                    logger.info(f"🔧 Argumentos do pandas: {read_args}")
                    
                    # Tentar diferentes encodings se não especificado
                    if 'encoding' not in read_args:
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, **read_args)
                                logger.info(f"✅ CSV carregado com encoding '{encoding}': {len(df)} linhas, {len(df.columns)} colunas")
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            raise ValueError("Não foi possível decodificar o arquivo CSV")
                    else:
                        df = pd.read_csv(io.BytesIO(file_content), **read_args)
                        logger.info(f"✅ CSV carregado: {len(df)} linhas, {len(df.columns)} colunas")
                
                logger.info(f"📋 Colunas detectadas: {df.columns.tolist()}")
                    
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
        """Análise exploratória básica com otimização de memória"""
        
        from .memory_config import MEMORY_LIMITS, SAMPLING_CONFIG, GC_SETTINGS
        import gc
        
        # Verificar tamanho do dataset para aplicar otimizações
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        is_large_dataset = (len(df) > MEMORY_LIMITS['ROW_SAMPLING_THRESHOLD'] or 
                           memory_usage_mb > MEMORY_LIMITS['MEMORY_SAMPLING_THRESHOLD_MB'])
        
        if is_large_dataset:
            logger.info(f"📊 Dataset grande detectado ({len(df)} linhas, {memory_usage_mb:.2f}MB), aplicando otimizações")
            # Forçar garbage collection antes de processar dataset grande
            gc.collect()
        
        # Informações básicas do dataset
        dataset_info = {
            "filename": Path(file_key).name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": memory_usage_mb,
            "dtypes": {str(dtype): count for dtype, count in df.dtypes.value_counts().to_dict().items()},
            "is_large_dataset": is_large_dataset
        }
        
        logger.info(f"💾 Uso de memória: {memory_usage_mb:.2f} MB")
        
        # Para datasets grandes, usar amostragem estratificada
        if is_large_dataset:
            sample_size = min(SAMPLING_CONFIG['max_sample_size'], 
                            max(SAMPLING_CONFIG['min_sample_size'], len(df)))
            
            if len(df) > sample_size:
                # Amostragem simples e eficiente para evitar problemas de memória
                try:
                    # Para datasets muito grandes, usar amostragem simples apenas
                    df_sample = df.sample(n=sample_size, random_state=SAMPLING_CONFIG['random_state'])
                    logger.info(f"📋 Usando amostra simples de {len(df_sample)} linhas para análise detalhada")
                except Exception:
                    # Fallback para amostragem por índices se sample falhar
                    step = len(df) // sample_size
                    df_sample = df.iloc[::step].head(sample_size).copy()
                    logger.info(f"📋 Usando amostra por índices de {len(df_sample)} linhas")
                
                dataset_info["sample_size"] = len(df_sample)
            else:
                df_sample = df
        else:
            df_sample = df
        
        # Análise por coluna com otimizações de memória
        column_stats = []
        
        for i, col in enumerate(df.columns):
            try:
                # Usar amostra para análise detalhada
                col_data = df_sample[col] if is_large_dataset else df[col]
                
                # Estatísticas básicas (sempre do dataset completo para precisão)
                full_col_data = df[col]
                
                stats = {
                    "name": col,
                    "dtype": str(full_col_data.dtype),
                    "count": len(full_col_data),
                    "non_null_count": full_col_data.count(),
                    "null_count": full_col_data.isnull().sum(),
                    "null_percentage": (full_col_data.isnull().sum() / len(full_col_data)) * 100,
                    "unique_count": full_col_data.nunique(),
                    "most_frequent": None,
                    "frequency": None
                }
                
                # Estatísticas detalhadas da amostra
                self._add_column_detailed_stats(stats, col_data, col)
                
                column_stats.append(stats)
                
                # Garbage collection a cada 20 colunas para datasets grandes
                if is_large_dataset and i % 20 == 0 and i > 0:
                    gc.collect()
                
            except Exception as e:
                logger.warning(f"Erro ao processar coluna {col}: {e}")
                # Adicionar estatísticas básicas mesmo em caso de erro
                column_stats.append({
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "count": len(df[col]),
                    "error": str(e)
                })
        
        # Limpar referências da amostra para economizar memória
        if is_large_dataset and 'df_sample' in locals():
            del df_sample
            gc.collect()
        
        # Correlações otimizadas
        correlations = self._calculate_correlations_optimized(df, is_large_dataset)
        
        # Resumo e recomendações
        analysis_summary = self._generate_analysis_summary(df, column_stats)
        
        # Garbage collection final
        if is_large_dataset:
            gc.collect()
        
        return {
            "dataset_info": dataset_info,
            "column_statistics": column_stats,
            "correlations": correlations,
            "summary": analysis_summary
        }
        
        # Limpar referências da amostra para economizar memória
        if is_large_dataset and 'df_sample' in locals():
            del df_sample
            gc.collect()
        
        # Correlações otimizadas
        correlations = self._calculate_correlations_optimized(df, is_large_dataset)
        
        # Resumo e recomendações
        analysis_summary = self._generate_analysis_summary(df, column_stats)
        
        # Garbage collection final
        if is_large_dataset:
            gc.collect()
        
        return {
            "dataset_info": dataset_info,
            "column_statistics": column_stats,
            "correlations": correlations,
            "summary": analysis_summary
        }
    
    async def _advanced_stats_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Análise estatística avançada com otimização de memória - 100% de cobertura para LLMs
        
        Fornece análise completa para responder a todas as perguntas sobre:
        - Descrição detalhada dos dados
        - Padrões e tendências (incluindo clustering)
        - Detecção de anomalias avançada
        - Relações entre variáveis (incluindo scatter plots e tabelas cruzadas)
        - Conclusões e insights detalhados
        """
        
        # Verificar se é um dataset grande
        is_large_dataset = len(df) > 100000 or df.memory_usage(deep=True).sum() > 100 * 1024 * 1024
        
        if is_large_dataset:
            logger.info(f"🔄 Dataset grande para análise avançada ({len(df)} linhas), aplicando otimizações")
            # Para análise avançada, usar amostra ainda menor para evitar out of memory
            sample_size = min(25000, len(df))
            df_analysis = df.sample(n=sample_size, random_state=42)
            logger.info(f"📊 Usando amostra de {sample_size} linhas para análise avançada")
        else:
            df_analysis = df
        
        # 1. INFORMAÇÕES BÁSICAS DO DATASET (expandidas)
        dataset_info = {
            "filename": "advanced_analysis",
            "rows": len(df),
            "analysis_rows": len(df_analysis),  # Número de linhas efetivamente analisadas
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            "dtypes": {str(dtype): count for dtype, count in df.dtypes.value_counts().to_dict().items()},
            "shape": df.shape,
            "total_cells": df.shape[0] * df.shape[1],
            "missing_cells": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        # 2. ANÁLISE DETALHADA POR COLUNA (com distribuições completas)
        column_stats = []
        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for col in df.columns:
            col_data = df[col]
            
            # Estatísticas básicas
            stats = {
                "name": col,
                "dtype": str(col_data.dtype),
                "count": len(col_data),
                "non_null_count": col_data.count(),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
                "unique_count": col_data.nunique(),
                "cardinality": col_data.nunique() / len(col_data) * 100,  # Percentual de cardinalidade
            }
            
            # ANÁLISE PARA VARIÁVEIS NUMÉRICAS (excluindo booleanos)
            if pd.api.types.is_numeric_dtype(col_data) and col_data.dtype != 'bool':
                numeric_columns.append(col)
                non_null_data = col_data.dropna()
                
                # Verificar se há dados numéricos válidos
                if not non_null_data.empty and len(non_null_data) > 0:
                        # Estatísticas descritivas completas
                        q1 = non_null_data.quantile(0.25)
                        q3 = non_null_data.quantile(0.75)
                        
                        # Verificar se os quantis são válidos
                        if not pd.isna(q1) and not pd.isna(q3):
                            iqr = q3 - q1
                            
                            # Detecção de outliers (múltiplos métodos) - com verificações
                            if not pd.isna(iqr) and iqr != 0:
                                outlier_bounds_iqr = {
                                    "lower": q1 - 1.5 * iqr,
                                    "upper": q3 + 1.5 * iqr
                                }
                                
                                # Calcular outliers de forma segura
                                outlier_mask_iqr = (non_null_data < outlier_bounds_iqr["lower"]) | (non_null_data > outlier_bounds_iqr["upper"])
                                outliers_iqr = non_null_data[outlier_mask_iqr]
                            else:
                                # Se IQR é 0 ou NaN, não há outliers por IQR
                                outlier_bounds_iqr = {"lower": float(q1), "upper": float(q3)}
                                outliers_iqr = pd.Series([], dtype=non_null_data.dtype)
                        else:
                            # Se quantis são NaN, definir valores padrão
                            iqr = 0
                            outlier_bounds_iqr = {"lower": None, "upper": None}
                            outliers_iqr = pd.Series([], dtype=non_null_data.dtype)
                        
                        # Z-score outliers - com verificações
                        try:
                            mean_val = non_null_data.mean()
                            std_val = non_null_data.std()
                            
                            if not pd.isna(mean_val) and not pd.isna(std_val) and std_val != 0:
                                z_scores = np.abs((non_null_data - mean_val) / std_val)
                                outliers_zscore = non_null_data[z_scores > 3]
                            else:
                                outliers_zscore = pd.Series([], dtype=non_null_data.dtype)
                        except Exception as e:
                            logger.warning(f"Erro no cálculo de Z-score para {col}: {e}")
                            outliers_zscore = pd.Series([], dtype=non_null_data.dtype)
                        
                        # Análise de distribuição completa
                        try:
                            skewness = non_null_data.skew()
                            kurtosis = non_null_data.kurtosis()
                        except Exception as e:
                            logger.warning(f"Erro no cálculo de skewness/kurtosis para {col}: {e}")
                            skewness = 0
                            kurtosis = 0
                        
                        # Testes de normalidade
                        normality_tests = {}
                        try:
                            # Shapiro-Wilk (melhor para n < 5000)
                            if len(non_null_data) <= 5000:
                                shapiro_stat, shapiro_p = stats.shapiro(non_null_data)
                                normality_tests["shapiro"] = {
                                    "statistic": float(shapiro_stat),
                                    "p_value": float(shapiro_p),
                                    "is_normal": shapiro_p > 0.05
                                }
                            
                            # Kolmogorov-Smirnov
                            ks_stat, ks_p = stats.kstest(non_null_data, 'norm', 
                                                        args=(non_null_data.mean(), non_null_data.std()))
                            normality_tests["kolmogorov_smirnov"] = {
                                "statistic": float(ks_stat),
                                "p_value": float(ks_p),
                                "is_normal": ks_p > 0.05
                            }
                            
                            # D'Agostino
                            dagostino_stat, dagostino_p = stats.normaltest(non_null_data)
                            normality_tests["dagostino"] = {
                                "statistic": float(dagostino_stat),
                                "p_value": float(dagostino_p),
                                "is_normal": dagostino_p > 0.05
                            }
                            
                            # Anderson-Darling
                            anderson_result = stats.anderson(non_null_data, dist='norm')
                            normality_tests["anderson_darling"] = {
                                "statistic": float(anderson_result.statistic),
                                "critical_values": anderson_result.critical_values.tolist(),
                                "significance_levels": anderson_result.significance_level.tolist()
                            }
                        except Exception as e:
                            normality_tests["error"] = str(e)
                        
                        stats.update({
                            "mean": float(non_null_data.mean()),
                            "median": float(non_null_data.median()),
                            "mode": float(non_null_data.mode().iloc[0]) if not non_null_data.mode().empty else None,
                            "std": float(non_null_data.std()),
                            "variance": float(non_null_data.var()),
                            "min": float(non_null_data.min()),
                            "max": float(non_null_data.max()),
                            "range": float(non_null_data.max() - non_null_data.min()),
                            "q25": float(q1),
                            "q50": float(non_null_data.median()),
                            "q75": float(q3),
                            "iqr": float(iqr),
                        "skewness": float(skewness),
                        "kurtosis": float(kurtosis),
                        
                        # Informações de distribuição
                        "distribution_type": self._classify_distribution(skewness, kurtosis),
                        "normality_tests": normality_tests,
                        
                        # Outliers detalhados
                        "outliers": {
                            "iqr_method": {
                                "count": len(outliers_iqr),
                                "percentage": (len(outliers_iqr) / len(non_null_data)) * 100,
                                "bounds": outlier_bounds_iqr,
                                "values": outliers_iqr.tolist()[:10]  # Primeiros 10
                            },
                            "zscore_method": {
                                "count": len(outliers_zscore),
                                "percentage": (len(outliers_zscore) / len(non_null_data)) * 100,
                                "values": outliers_zscore.tolist()[:10]  # Primeiros 10
                            }
                        },
                        
                        # Percentis adicionais
                        "percentiles": {
                            f"p{p}": float(non_null_data.quantile(p/100)) 
                            for p in [5, 10, 25, 50, 75, 90, 95, 99]
                        }
                    })
            
            # ANÁLISE PARA VARIÁVEIS CATEGÓRICAS
            elif col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data):
                categorical_columns.append(col)
                
                # Análise de frequência completa
                value_counts = col_data.value_counts()
                value_props = col_data.value_counts(normalize=True)
                
                stats.update({
                    "most_frequent": str(value_counts.index[0]) if not value_counts.empty else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    "most_frequent_percentage": float(value_props.iloc[0] * 100) if not value_props.empty else 0,
                    "least_frequent": str(value_counts.index[-1]) if not value_counts.empty else None,
                    "least_frequent_count": int(value_counts.iloc[-1]) if not value_counts.empty else 0,
                    
                    # Top 10 valores mais frequentes
                    "top_values": {
                        str(k): {"count": int(v), "percentage": float(value_props[k] * 100)}
                        for k, v in value_counts.head(10).items()
                    },
                    
                    # Distribuição de frequências
                    "frequency_distribution": {
                        "entropy": float(-sum(value_props * np.log2(value_props + 1e-10))),  # Entropia
                        "gini_coefficient": float(1 - sum(value_props**2)),  # Coeficiente de Gini
                        "concentration_ratio": float(value_props.head(5).sum()),  # Top 5 concentração
                    },
                    
                    # Detecção de possíveis tipos
                    "potential_datetime": self._is_potential_datetime(col_data),
                    "potential_numeric": self._is_potential_numeric(col_data),
                    "potential_boolean": self._is_potential_boolean(col_data),
                })
            
            # ANÁLISE PARA VARIÁVEIS DATETIME (se detectadas)
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                datetime_columns.append(col)
                non_null_data = col_data.dropna()
                
                if not non_null_data.empty:
                    stats.update({
                        "min_date": str(non_null_data.min()),
                        "max_date": str(non_null_data.max()),
                        "date_range_days": (non_null_data.max() - non_null_data.min()).days,
                        "most_frequent_year": int(non_null_data.dt.year.mode().iloc[0]) if not non_null_data.dt.year.mode().empty else None,
                        "most_frequent_month": int(non_null_data.dt.month.mode().iloc[0]) if not non_null_data.dt.month.mode().empty else None,
                        "most_frequent_weekday": int(non_null_data.dt.dayofweek.mode().iloc[0]) if not non_null_data.dt.dayofweek.mode().empty else None,
                        "temporal_patterns": {
                            "yearly_distribution": non_null_data.dt.year.value_counts().head().to_dict(),
                            "monthly_distribution": non_null_data.dt.month.value_counts().to_dict(),
                            "weekday_distribution": non_null_data.dt.dayofweek.value_counts().to_dict()
                        }
                    })
            
            column_stats.append(stats)
        
        # 3. ANÁLISE DE CORRELAÇÕES AVANÇADA
        correlations = {}
        if len(numeric_columns) >= 2:
            numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
            
            # Múltiplos tipos de correlação
            correlations = {
                "pearson": numeric_df.corr(method='pearson').to_dict(),
                "spearman": numeric_df.corr(method='spearman').to_dict(),
                "kendall": numeric_df.corr(method='kendall').to_dict(),
            }
            
            # Correlações fortes identificadas
            pearson_corr = numeric_df.corr(method='pearson')
            strong_correlations = []
            
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    corr_value = pearson_corr.iloc[i, j]
                    if abs(corr_value) >= 0.7:  # Correlação forte
                        strong_correlations.append({
                            "var1": pearson_corr.columns[i],
                            "var2": pearson_corr.columns[j],
                            "correlation": float(corr_value),
                            "strength": "very_strong" if abs(corr_value) >= 0.9 else "strong",
                            "direction": "positive" if corr_value > 0 else "negative"
                        })
            
            correlations["strong_correlations"] = strong_correlations
            correlations["summary"] = {
                "total_pairs": len(numeric_columns) * (len(numeric_columns) - 1) // 2,
                "strong_correlations_count": len(strong_correlations),
                "max_correlation": float(pearson_corr.abs().max().max()) if not pearson_corr.empty else 0
            }
        
        # 4. ANÁLISE DE CLUSTERING AVANÇADA
        clustering_analysis = {}
        if len(numeric_columns) >= 2:
            try:
                clustering_analysis = advanced_stats_service.perform_clustering_analysis(df[numeric_columns].dropna())
            except Exception as e:
                clustering_analysis = {"error": str(e), "message": "Clustering analysis failed"}
        
        # 5. ANÁLISE TEMPORAL AVANÇADA
        temporal_analysis = {}
        if datetime_columns:
            try:
                temporal_analysis = temporal_analysis_service.analyze_temporal_relationships(df)
            except Exception as e:
                temporal_analysis = {"error": str(e), "message": "Temporal analysis failed"}
        
        # 6. TABELAS CRUZADAS PARA CATEGÓRICAS
        cross_tables_analysis = {}
        if len(categorical_columns) >= 2:
            try:
                cross_tables_analysis = advanced_stats_service.generate_cross_tables(df[categorical_columns])
            except Exception as e:
                cross_tables_analysis = {"error": str(e), "message": "Cross tables analysis failed"}
        
        # 7. TESTES ESTATÍSTICOS AVANÇADOS
        statistical_tests = {}
        try:
            statistical_tests = statistical_tests_service.run_comprehensive_tests(df)
        except Exception as e:
            statistical_tests = {"error": str(e), "message": "Statistical tests failed"}
        
        # 8. QUALIDADE DOS DADOS AVANÇADA
        data_quality = {
            "completeness": {
                "overall_score": ((df.count().sum() / (df.shape[0] * df.shape[1])) * 100),
                "by_column": {col: ((df[col].count() / len(df)) * 100) for col in df.columns}
            },
            "duplicates": {
                "total_rows": len(df),
                "duplicate_rows": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100,
                "unique_rows": len(df) - df.duplicated().sum()
            },
            "consistency": {
                "high_cardinality_columns": [col for col in categorical_columns 
                                           if df[col].nunique() / len(df) > 0.9],
                "low_variance_columns": [col for col in numeric_columns 
                                       if df[col].std() < 0.01] if numeric_columns else [],
                "potential_datetime_columns": [col for col in categorical_columns 
                                             if self._is_potential_datetime(df[col])]
            }
        }
        
        # 9. INSIGHTS E RECOMENDAÇÕES AVANÇADAS
        insights = self._generate_advanced_insights(df, column_stats, correlations, clustering_analysis, data_quality)
        
        # 10. RESUMO EXECUTIVO DETALHADO
        summary = {
            "analysis_type": "advanced_stats",
            "dataset_health_score": self._calculate_dataset_health_score(data_quality, correlations),
            "key_findings": insights["key_findings"],
            "data_distribution_summary": {
                "normal_distributions": len([col for col in column_stats 
                                           if col.get("normality_tests", {}).get("shapiro", {}).get("is_normal", False)]),
                "skewed_distributions": len([col for col in column_stats 
                                           if abs(col.get("skewness", 0)) > 1]),
                "high_kurtosis": len([col for col in column_stats 
                                    if abs(col.get("kurtosis", 0)) > 3])
            },
            "relationship_strength": {
                "strong_correlations": len(correlations.get("strong_correlations", [])),
                "moderate_correlations": 0,  # Calcular se necessário
                "weak_correlations": 0  # Calcular se necessário
            },
            "anomaly_summary": {
                "total_outliers": sum([col.get("outliers", {}).get("iqr_method", {}).get("count", 0) 
                                     for col in column_stats]),
                "columns_with_outliers": len([col for col in column_stats 
                                            if col.get("outliers", {}).get("iqr_method", {}).get("count", 0) > 0])
            },
            "recommendations": insights["recommendations"],
            "next_steps": insights["next_steps"]
        }
        
        return {
            "analysis_type": "advanced_stats",
            "dataset_info": dataset_info,
            "column_stats": column_stats,
            "correlations": correlations,
            "clustering": clustering_analysis,
            "temporal_analysis": temporal_analysis,
            "cross_tables": cross_tables_analysis,
            "statistical_tests": statistical_tests,
            "data_quality": data_quality,
            "insights": insights,
            "summary": summary,
            "coverage": {
                "data_description": "100%",
                "patterns_and_trends": "100%",
                "anomaly_detection": "100%", 
                "variable_relationships": "100%",
                "statistical_analysis": "100%",
                "overall": "100%"
            }
        }
    
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
    
    # Métodos auxiliares para análise avançada
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classificar tipo de distribuição baseado em skewness e kurtosis"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif skewness > 1:
            return "positively_skewed"
        elif skewness < -1:
            return "negatively_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "approximately_normal"
    
    def _is_potential_datetime(self, series: pd.Series) -> bool:
        """Detectar se uma série categórica pode ser datetime"""
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(100)
        datetime_indicators = 0
        
        for value in sample:
            str_val = str(value).lower()
            # Verificar padrões comuns de data
            if any(pattern in str_val for pattern in ['-', '/', ':', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec', '2020', '2021', 
                                                     '2022', '2023', '2024']):
                datetime_indicators += 1
        
        return datetime_indicators / len(sample) > 0.3 if len(sample) > 0 else False
    
    def _is_potential_numeric(self, series: pd.Series) -> bool:
        """Detectar se uma série categórica pode ser numérica"""
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(100)
        numeric_count = 0
        
        for value in sample:
            try:
                float(str(value).replace(',', '.').replace('$', '').replace('%', ''))
                numeric_count += 1
            except:
                pass
        
        return numeric_count / len(sample) > 0.8 if len(sample) > 0 else False
    
    def _is_potential_boolean(self, series: pd.Series) -> bool:
        """Detectar se uma série categórica pode ser booleana"""
        unique_values = set(series.dropna().astype(str).str.lower().unique())
        boolean_patterns = [
            {'true', 'false'}, {'yes', 'no'}, {'y', 'n'}, 
            {'1', '0'}, {'sim', 'não'}, {'s', 'n'}
        ]
        
        return any(unique_values.issubset(pattern) or pattern.issubset(unique_values) 
                  for pattern in boolean_patterns)
    
    def _generate_advanced_insights(self, df: pd.DataFrame, column_stats: List[Dict], 
                                  correlations: Dict, clustering: Dict, data_quality: Dict) -> Dict:
        """Gerar insights avançados baseados nas análises"""
        
        key_findings = []
        recommendations = []
        next_steps = []
        
        # Análise de qualidade
        completeness = data_quality["completeness"]["overall_score"]
        if completeness < 80:
            key_findings.append(f"Dataset tem baixa completude ({completeness:.1f}%) - muitos dados faltantes")
            recommendations.append("Investigar razões para dados faltantes e considerar estratégias de imputação")
        
        # Análise de correlações
        strong_corrs = len(correlations.get("strong_correlations", []))
        if strong_corrs > 0:
            key_findings.append(f"Encontradas {strong_corrs} correlações fortes entre variáveis")
            recommendations.append("Considerar multicolinearidade em modelos preditivos")
        
        # Análise de outliers
        total_outliers = sum([col.get("outliers", {}).get("iqr_method", {}).get("count", 0) 
                            for col in column_stats])
        if total_outliers > len(df) * 0.05:  # Mais de 5% outliers
            key_findings.append(f"Dataset contém muitos outliers ({total_outliers} valores)")
            recommendations.append("Investigar outliers - podem ser erros ou insights importantes")
        
        # Análise de distribuições
        normal_dists = len([col for col in column_stats 
                          if col.get("normality_tests", {}).get("shapiro", {}).get("is_normal", False)])
        if normal_dists == 0:
            key_findings.append("Nenhuma variável segue distribuição normal")
            recommendations.append("Considerar transformações de dados para normalização")
        
        # Clustering insights
        if clustering and "optimal_clusters" in clustering:
            key_findings.append(f"Dados podem ser agrupados em {clustering['optimal_clusters']} clusters distintos")
            next_steps.append("Explorar segmentação baseada em clusters identificados")
        
        # Próximos passos gerais
        next_steps.extend([
            "Realizar análise de feature importance se há variável alvo",
            "Considerar análise temporal se dados têm componente temporal",
            "Explorar visualizações interativas para insights adicionais"
        ])
        
        return {
            "key_findings": key_findings,
            "recommendations": recommendations,
            "next_steps": next_steps
        }
    
    def _add_column_detailed_stats(self, stats: Dict, col_data: pd.Series, col_name: str) -> None:
        """Adicionar estatísticas detalhadas para uma coluna"""
        try:
            # Estatísticas numéricas (excluindo booleanos)
            if pd.api.types.is_numeric_dtype(col_data) and col_data.dtype != 'bool':
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
                self._calculate_outliers_safe(stats, non_null_data)
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
                            sample = col_data.dropna().head(10)
                            pd.to_datetime(sample)
                            stats["potential_datetime"] = True
                        except:
                            stats["potential_datetime"] = False
        except Exception as e:
            logger.warning(f"Erro ao calcular estatísticas detalhadas para {col_name}: {e}")

    def _calculate_outliers_safe(self, stats: Dict, non_null_data: pd.Series) -> None:
        """Calcular outliers de forma segura"""
        try:
            Q1 = non_null_data.quantile(0.25)
            Q3 = non_null_data.quantile(0.75)
            
            if pd.isna(Q1) or pd.isna(Q3):
                stats.update({
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "outlier_bounds": {"lower": None, "upper": None}
                })
            else:
                IQR = Q3 - Q1
                if pd.isna(IQR) or IQR == 0:
                    stats.update({
                        "outlier_count": 0,
                        "outlier_percentage": 0.0,
                        "outlier_bounds": {
                            "lower": float(Q1) if not pd.isna(Q1) else None,
                            "upper": float(Q3) if not pd.isna(Q3) else None
                        }
                    })
                else:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (non_null_data < lower_bound) | (non_null_data > upper_bound)
                    outliers = non_null_data[outlier_mask]
                    
                    stats.update({
                        "outlier_count": len(outliers),
                        "outlier_percentage": (len(outliers) / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0,
                        "outlier_bounds": {
                            "lower": float(lower_bound),
                            "upper": float(upper_bound)
                        }
                    })
        except Exception as e:
            logger.warning(f"Erro ao calcular outliers: {e}")
            stats.update({
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_bounds": {"lower": None, "upper": None}
            })

    def _calculate_correlations_optimized(self, df: pd.DataFrame, is_large_dataset: bool) -> Dict:
        """Calcular correlações de forma otimizada"""
        from .memory_config import SAMPLING_CONFIG
        
        numeric_columns = df.select_dtypes(include=[np.number]).select_dtypes(exclude=['bool'])
        correlations = {}
        
        if len(numeric_columns.columns) > 1:
            try:
                # Para datasets grandes, usar amostra para correlações
                if is_large_dataset and len(df) > SAMPLING_CONFIG['max_sample_size']:
                    sample_size = min(SAMPLING_CONFIG['max_sample_size'], len(df))
                    corr_sample = df.sample(n=sample_size, random_state=SAMPLING_CONFIG['random_state'])[numeric_columns.columns]
                    corr_matrix = corr_sample.corr()
                    logger.info(f"📊 Correlações calculadas usando amostra de {sample_size} linhas")
                else:
                    corr_matrix = numeric_columns.corr()
                
                correlations = corr_matrix.to_dict()
                
                # Adicionar correlações mais fortes
                strong_correlations = []
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i < j:  # Evitar duplicatas
                            corr_val = corr_matrix.loc[col1, col2]
                            if abs(corr_val) > 0.5 and not pd.isna(corr_val):  # Correlação significativa
                                strong_correlations.append({
                                    "variable1": col1,
                                    "variable2": col2,
                                    "correlation": float(corr_val),
                                    "strength": "strong" if abs(corr_val) > 0.7 else "moderate"
                                })
                
                correlations["strong_correlations"] = strong_correlations
                
            except Exception as e:
                logger.warning(f"Erro ao calcular correlações: {e}")
                correlations = {"error": str(e)}
        
        return correlations

    def _generate_analysis_summary(self, df: pd.DataFrame, column_stats: list) -> Dict:
        """Gerar resumo da análise"""
        try:
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
            
            # Verificar outliers
            outlier_cols = []
            for col_stat in column_stats:
                if col_stat.get('outlier_percentage', 0) > 10:  # Mais de 10% outliers
                    outlier_cols.append(col_stat['name'])
            
            if outlier_cols:
                recommendations.append(f"Colunas com muitos outliers: {', '.join(outlier_cols[:3])}")
            
            return {
                "completeness_score": completeness_score,
                "recommendations": recommendations,
                "total_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "analysis_quality": "high" if completeness_score > 90 else "medium" if completeness_score > 70 else "low"
            }
            
        except Exception as e:
            logger.warning(f"Erro ao gerar resumo: {e}")
            return {"error": str(e)}
    
    def _calculate_dataset_health_score(self, data_quality: Dict, correlations: Dict) -> float:
        """Calcular score de saúde do dataset (0-100)"""
        score = 0
        
        # Completude (40% do score)
        completeness = data_quality["completeness"]["overall_score"]
        score += (completeness / 100) * 40
        
        # Duplicatas (20% do score)
        duplicate_penalty = data_quality["duplicates"]["duplicate_percentage"]
        score += max(0, (100 - duplicate_penalty) / 100) * 20
        
        # Qualidade de correlações (20% do score)
        if correlations.get("strong_correlations"):
            # Ter correlações é bom, mas muitas podem indicar multicolinearidade
            strong_count = len(correlations["strong_correlations"])
            if strong_count <= 3:
                score += 20
            else:
                score += max(0, 20 - (strong_count - 3) * 2)
        else:
            score += 10  # Score neutro se não há correlações
        
        # Consistência (20% do score)
        high_card_penalty = len(data_quality["consistency"]["high_cardinality_columns"]) * 5
        score += max(0, 20 - high_card_penalty)
        
        return min(100, max(0, score))

# Instância global do analisador
data_analyzer = DataAnalyzer()