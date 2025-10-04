"""
Serviço de geração de visualizações para análise EDA
"""
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para não usar GUI
plt.switch_backend('Agg')
sns.set_style("whitegrid")

class VisualizationService:
    """Serviço para geração de visualizações de dados"""
    
    def __init__(self):
        self.figure_size = (10, 6)
        self.dpi = 100
    
    def generate_histogram(self, df: pd.DataFrame, column: str) -> str:
        """
        Gerar histograma para variável numérica
        
        Args:
            df: DataFrame
            column: Nome da coluna
            
        Returns:
            Histograma em base64
        """
        try:
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Remover valores nulos
            data = df[column].dropna()
            
            if len(data) == 0:
                return self._empty_plot("Sem dados para exibir")
            
            # Criar histograma
            plt.hist(data, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribuição de {column}')
            plt.xlabel(column)
            plt.ylabel('Frequência')
            
            # Adicionar linha de média
            mean_val = data.mean()
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
            plt.legend()
            
            plt.tight_layout()
            
            # Converter para base64
            return self._plot_to_base64()
            
        except Exception as e:
            return self._empty_plot(f"Erro: {str(e)}")
    
    def generate_boxplot(self, df: pd.DataFrame, column: str) -> str:
        """
        Gerar boxplot para visualizar outliers
        
        Args:
            df: DataFrame
            column: Nome da coluna
            
        Returns:
            Boxplot em base64
        """
        try:
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            data = df[column].dropna()
            
            if len(data) == 0:
                return self._empty_plot("Sem dados para exibir")
            
            plt.boxplot(data, vert=True)
            plt.title(f'Boxplot de {column}')
            plt.ylabel(column)
            
            plt.tight_layout()
            return self._plot_to_base64()
            
        except Exception as e:
            return self._empty_plot(f"Erro: {str(e)}")
    
    def generate_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """
        Gerar scatter plot entre duas variáveis numéricas
        
        Args:
            df: DataFrame
            x_col: Coluna do eixo X
            y_col: Coluna do eixo Y
            
        Returns:
            Scatter plot em base64
        """
        try:
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Remover linhas com valores nulos
            clean_data = df[[x_col, y_col]].dropna()
            
            if len(clean_data) == 0:
                return self._empty_plot("Sem dados para exibir")
            
            plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, color='skyblue')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{y_col} vs {x_col}')
            
            # Adicionar linha de tendência
            try:
                z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                p = np.poly1d(z)
                plt.plot(clean_data[x_col], p(clean_data[x_col]), "r--", alpha=0.8)
                
                # Calcular correlação
                corr = clean_data[x_col].corr(clean_data[y_col])
                plt.text(0.05, 0.95, f'Correlação: {corr:.3f}', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            except:
                pass
            
            plt.tight_layout()
            return self._plot_to_base64()
            
        except Exception as e:
            return self._empty_plot(f"Erro: {str(e)}")
    
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """
        Gerar heatmap de correlação
        
        Args:
            df: DataFrame
            
        Returns:
            Heatmap em base64
        """
        try:
            # Selecionar apenas colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number])
            
            if numeric_cols.empty:
                return self._empty_plot("Nenhuma variável numérica encontrada")
            
            plt.figure(figsize=(12, 8), dpi=self.dpi)
            
            # Calcular correlação
            corr_matrix = numeric_cols.corr()
            
            # Criar heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, cbar_kws={'label': 'Correlação'})
            
            plt.title('Matriz de Correlação')
            plt.tight_layout()
            
            return self._plot_to_base64()
            
        except Exception as e:
            return self._empty_plot(f"Erro: {str(e)}")
    
    def generate_categorical_bar_chart(self, df: pd.DataFrame, column: str, top_n: int = 10) -> str:
        """
        Gerar gráfico de barras para variável categórica
        
        Args:
            df: DataFrame
            column: Nome da coluna categórica
            top_n: Número de categorias principais a exibir
            
        Returns:
            Gráfico de barras em base64
        """
        try:
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Contar frequências
            value_counts = df[column].value_counts().head(top_n)
            
            if len(value_counts) == 0:
                return self._empty_plot("Sem dados para exibir")
            
            # Criar gráfico de barras
            value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title(f'Distribuição de {column} (Top {len(value_counts)})')
            plt.xlabel(column)
            plt.ylabel('Frequência')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            return self._plot_to_base64()
            
        except Exception as e:
            return self._empty_plot(f"Erro: {str(e)}")
    
    def generate_cross_table_heatmap(self, df: pd.DataFrame, col1: str, col2: str) -> str:
        """
        Gerar heatmap de tabela cruzada entre duas variáveis categóricas
        
        Args:
            df: DataFrame
            col1: Primeira variável categórica
            col2: Segunda variável categórica
            
        Returns:
            Heatmap da tabela cruzada em base64
        """
        try:
            plt.figure(figsize=(12, 8), dpi=self.dpi)
            
            # Criar tabela cruzada
            cross_tab = pd.crosstab(df[col1], df[col2])
            
            if cross_tab.empty:
                return self._empty_plot("Sem dados para tabela cruzada")
            
            # Criar heatmap
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Frequência'})
            plt.title(f'Tabela Cruzada: {col1} vs {col2}')
            plt.xlabel(col2)
            plt.ylabel(col1)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            return self._plot_to_base64()
            
        except Exception as e:
            return self._empty_plot(f"Erro: {str(e)}")
    
    def _plot_to_base64(self) -> str:
        """Converter plot atual para base64"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.dpi)
        buffer.seek(0)
        
        # Converter para base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Limpar plot da memória
        
        return f"data:image/png;base64,{image_base64}"
    
    def _empty_plot(self, message: str) -> str:
        """Criar plot vazio com mensagem"""
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.text(0.5, 0.5, message, ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, bbox=dict(boxstyle="round", facecolor='lightgray'))
        plt.axis('off')
        plt.tight_layout()
        return self._plot_to_base64()

# Instância global do serviço
visualization_service = VisualizationService()