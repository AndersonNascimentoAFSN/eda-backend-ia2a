"""
Configurações de otimização de memória para processamento de grandes datasets
"""

# Limites de memória para diferentes operações
MEMORY_LIMITS = {
    # Tamanho máximo do arquivo em MB para usar leitura em chunks
    'CHUNK_THRESHOLD_MB': 50,
    
    # Tamanho do chunk para leitura de arquivos grandes
    'CHUNK_SIZE': 10000,
    
    # Tamanho máximo da amostra para detecção de separador CSV (bytes)
    'CSV_SEPARATOR_SAMPLE_SIZE': 50 * 1024,  # 50KB
    
    # Tamanho máximo da amostra para análise de dtypes
    'DTYPE_ANALYSIS_SAMPLE_SIZE': 1000,
    
    # Tamanho máximo da amostra para análises estatísticas detalhadas
    'STATS_SAMPLE_SIZE': 50000,
    
    # Limite de memória para usar amostragem (MB)
    'MEMORY_SAMPLING_THRESHOLD_MB': 100,
    
    # Limite de linhas para usar amostragem
    'ROW_SAMPLING_THRESHOLD': 100000,
}

# Configurações de otimização de tipos de dados
DTYPE_OPTIMIZATIONS = {
    # Mapeamento de int64 para tipos menores baseado no range
    'int_optimization': {
        'int16_range': (-32768, 32767),
        'int32_range': (-2147483648, 2147483647),
    },
    
    # Conversão float64 -> float32
    'float_optimization': True,
    
    # Conversão para categoria baseada na cardinalidade
    'category_threshold': 0.5,  # Se nunique/len < 0.5, converter para category
}

# Configurações de garbage collection
GC_SETTINGS = {
    # Intervalo de chunks para executar garbage collection
    'gc_every_n_chunks': 10,
    
    # Forçar coleta de lixo após operações de memória intensiva
    'force_gc_after_operations': ['chunk_concat', 'dtype_conversion', 'sampling'],
}

# Configurações de amostragem para análises
SAMPLING_CONFIG = {
    # Amostra estratificada para datasets grandes
    'stratified_sampling': True,
    
    # Semente aleatória para reprodutibilidade
    'random_state': 42,
    
    # Tamanho mínimo da amostra
    'min_sample_size': 1000,
    
    # Tamanho máximo da amostra
    'max_sample_size': 50000,
}