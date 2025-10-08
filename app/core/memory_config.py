"""
Configurações de otimização de memória para o EDA Backend
"""

# Limites de memória para diferentes tipos de análise
MEMORY_LIMITS = {
    "small_dataset": 50 * 1024 * 1024,  # 50MB
    "large_dataset": 100 * 1024 * 1024,  # 100MB
    "max_rows_full_analysis": 100000,    # Máximo de linhas para análise completa
    "max_rows_sample": 50000,            # Tamanho da amostra para datasets grandes
    "max_rows_advanced_sample": 25000,   # Amostra para análise avançada
    "chunk_size": 10000,                 # Tamanho dos chunks para leitura
    "max_columns_correlation": 50,       # Máximo de colunas para correlação
    "outlier_sample_size": 10000,        # Amostra para detecção de outliers
}

# Configurações de otimização de tipos
DTYPE_OPTIMIZATIONS = {
    "int64_to_int32_threshold": 2**31 - 1,
    "int64_to_int16_threshold": 2**15 - 1,
    "float64_to_float32": True,
    "object_to_category_threshold": 0.5,  # % de valores únicos
}

# Configurações de garbage collection
GC_SETTINGS = {
    "force_gc_after_chunk": True,
    "force_gc_after_analysis": True,
    "gc_threshold_mb": 100,  # Forçar GC após usar mais de 100MB
}