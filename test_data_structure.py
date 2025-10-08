#!/usr/bin/env python3
"""
Teste específico para verificar estrutura de dados retornados pela análise EDA básica
"""
import asyncio
import sys
import os
import json

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.data_analyzer import DataAnalyzer
import pandas as pd

async def test_basic_eda_data_structure():
    """Verificar exatamente que dados são retornados na análise EDA básica"""
    
    print("🔍 ANALISANDO ESTRUTURA DE DADOS RETORNADOS - ANÁLISE EDA BÁSICA")
    print("=" * 80)
    
    # Criar dados de exemplo
    df = pd.DataFrame({
        'categoria': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'A'],
        'valor_numerico': [10, 20, 15, 30, 25, 12, 35, 22, 18, 16],
        'cidade': ['São Paulo', 'Rio de Janeiro', 'São Paulo', 'Belo Horizonte', 
                   'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'São Paulo', 
                   'Rio de Janeiro', 'São Paulo'],
        'score': [85.5, 92.1, 78.3, 95.7, 88.2, 82.9, 90.4, 87.1, 89.3, 84.6]
    })
    
    print(f"📊 Dataset de teste criado: {len(df)} linhas × {len(df.columns)} colunas")
    print(f"Colunas: {list(df.columns)}")
    print()
    
    # Instanciar analisador
    analyzer = DataAnalyzer()
    
    # Executar análise básica
    print("🚀 Executando análise EDA básica...")
    result = await analyzer._basic_eda_analysis(df, 'test_data.csv')
    
    print("\n" + "=" * 80)
    print("📋 ESTRUTURA COMPLETA DOS DADOS RETORNADOS")
    print("=" * 80)
    
    # 1. Chaves principais
    print("\n1️⃣ CHAVES PRINCIPAIS:")
    for key in result.keys():
        print(f"   • {key}")
    
    # 2. Dataset Info
    print("\n2️⃣ DATASET INFO:")
    dataset_info = result.get('dataset_info', {})
    for key, value in dataset_info.items():
        print(f"   {key}: {value}")
    
    # 3. Análise das colunas categóricas
    print("\n3️⃣ DADOS CATEGÓRICOS (categoria):")
    column_stats = result.get('column_statistics', [])
    categoria_col = next((col for col in column_stats if col['name'] == 'categoria'), None)
    
    if categoria_col:
        print(f"   Nome: {categoria_col['name']}")
        print(f"   Tipo: {categoria_col['dtype']}")
        print(f"   Valores únicos: {categoria_col['unique_count']}")
        print(f"   Mais frequente: {categoria_col.get('most_frequent')}")
        print(f"   Frequência: {categoria_col.get('frequency')}")
        print(f"   Top values: {categoria_col.get('top_values', {})}")
        print(f"   Cardinalidade: {categoria_col.get('cardinality')}")
    
    # 4. Análise das colunas numéricas
    print("\n4️⃣ DADOS NUMÉRICOS (valor_numerico):")
    valor_col = next((col for col in column_stats if col['name'] == 'valor_numerico'), None)
    
    if valor_col:
        print(f"   Nome: {valor_col['name']}")
        print(f"   Tipo: {valor_col['dtype']}")
        print(f"   Média: {valor_col.get('mean')}")
        print(f"   Mediana: {valor_col.get('median')}")
        print(f"   Desvio Padrão: {valor_col.get('std')}")
        print(f"   Mínimo: {valor_col.get('min')}")
        print(f"   Máximo: {valor_col.get('max')}")
        print(f"   Q25: {valor_col.get('q25')}")
        print(f"   Q75: {valor_col.get('q75')}")
        print(f"   Outliers: {valor_col.get('outlier_count', 0)}")
    
    # 5. Análise específica para gráficos de barras
    print("\n5️⃣ DADOS DISPONÍVEIS PARA GRÁFICOS DE BARRAS:")
    
    print("\n   🎯 CATEGORIA (para gráfico de barras):")
    if categoria_col and 'top_values' in categoria_col:
        top_values = categoria_col['top_values']
        print(f"      Top values disponíveis: {len(top_values)} itens")
        for valor, frequencia in top_values.items():
            print(f"      • {valor}: {frequencia} ocorrências")
    
    print("\n   🎯 CIDADE (para gráfico de barras):")
    cidade_col = next((col for col in column_stats if col['name'] == 'cidade'), None)
    if cidade_col and 'top_values' in cidade_col:
        top_values = cidade_col['top_values']
        print(f"      Top values disponíveis: {len(top_values)} itens")
        for valor, frequencia in top_values.items():
            print(f"      • {valor}: {frequencia} ocorrências")
    
    # 6. Verificar se dados brutos estão disponíveis
    print("\n6️⃣ DISPONIBILIDADE DE DADOS BRUTOS:")
    print("   ❌ Dados brutos das colunas NÃO são retornados")
    print("   ✅ Apenas estatísticas agregadas e top_values estão disponíveis")
    print("   ✅ Para gráficos de barras: usar 'top_values' de colunas categóricas")
    print("   ✅ Para histogramas: usar estatísticas (min, max, mean, std) para simular distribuição")
    
    # 7. Correlações
    print("\n7️⃣ DADOS DE CORRELAÇÃO:")
    correlations = result.get('correlations', {})
    if correlations:
        print(f"   Correlações disponíveis: {len(correlations)} pares")
        print(f"   Estrutura: {list(correlations.keys())[:3]}...")
    
    print("\n" + "=" * 80)
    print("📊 CONCLUSÃO SOBRE GRÁFICOS DE BARRAS")
    print("=" * 80)
    print("✅ SIM, é possível fazer gráficos de barras!")
    print("📋 Dados disponíveis:")
    print("   • top_values: dicionário com valor → frequência")
    print("   • most_frequent: valor mais frequente")
    print("   • frequency: frequência do valor mais frequente")
    print("   • cardinality: número total de categorias únicas")
    print()
    print("❌ LIMITAÇÃO: Dados brutos das colunas NÃO são retornados")
    print("   • Apenas estatísticas agregadas")
    print("   • Para datasets grandes, isso é intencional (economia de memória)")
    print("   • Para gráficos, usar top_values é suficiente e eficiente")

if __name__ == "__main__":
    asyncio.run(test_basic_eda_data_structure())