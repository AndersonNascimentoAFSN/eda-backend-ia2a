"""
Serviço de análise temporal avançada
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import re
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class TemporalAnalysisService:
    """Serviço para análise temporal avançada"""
    
    def __init__(self):
        self.datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',           # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',           # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}',           # YYYY/MM/DD
            r'\d{2}/\d{2}/\d{2}',           # DD/MM/YY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]
    
    def detect_temporal_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detectar colunas temporais no DataFrame
        
        Args:
            df: DataFrame para análise
            
        Returns:
            Informações sobre colunas temporais detectadas
        """
        try:
            temporal_columns = {}
            
            for col in df.columns:
                analysis = self._analyze_column_for_temporal_patterns(df[col])
                if analysis["is_temporal"]:
                    temporal_columns[col] = analysis
            
            return {
                "temporal_columns": temporal_columns,
                "total_temporal_columns": len(temporal_columns),
                "recommendations": self._generate_temporal_recommendations(temporal_columns)
            }
            
        except Exception as e:
            return {"error": f"Erro na detecção temporal: {str(e)}"}
    
    def analyze_time_series(self, df: pd.DataFrame, date_column: str, 
                           value_column: str) -> Dict[str, Any]:
        """
        Análise detalhada de série temporal
        
        Args:
            df: DataFrame
            date_column: Nome da coluna de data
            value_column: Nome da coluna de valores
            
        Returns:
            Análise completa da série temporal
        """
        try:
            # Preparar dados
            df_temp = df[[date_column, value_column]].copy()
            df_temp = df_temp.dropna()
            
            # Converter coluna de data
            df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
            df_temp = df_temp.dropna()
            
            if len(df_temp) < 3:
                return {"error": "Dados insuficientes para análise temporal"}
            
            # Ordenar por data
            df_temp = df_temp.sort_values(date_column)
            df_temp = df_temp.reset_index(drop=True)
            
            # Análise básica
            time_span = df_temp[date_column].max() - df_temp[date_column].min()
            frequency = self._detect_frequency(df_temp[date_column])
            
            # Análise de tendência
            trend_analysis = self._analyze_trend(df_temp, date_column, value_column)
            
            # Análise de sazonalidade
            seasonality_analysis = self._analyze_seasonality(df_temp, date_column, value_column)
            
            # Estatísticas temporais
            temporal_stats = self._calculate_temporal_stats(df_temp, value_column)
            
            # Detecção de anomalias temporais
            anomalies = self._detect_temporal_anomalies(df_temp, value_column)
            
            # Padrões temporais
            patterns = self._identify_temporal_patterns(df_temp, date_column, value_column)
            
            return {
                "time_series_info": {
                    "start_date": df_temp[date_column].min().isoformat(),
                    "end_date": df_temp[date_column].max().isoformat(),
                    "time_span_days": time_span.days,
                    "total_observations": len(df_temp),
                    "estimated_frequency": frequency
                },
                "trend_analysis": trend_analysis,
                "seasonality_analysis": seasonality_analysis,
                "temporal_statistics": temporal_stats,
                "anomalies": anomalies,
                "patterns": patterns,
                "recommendations": self._generate_time_series_recommendations(
                    trend_analysis, seasonality_analysis, anomalies
                )
            }
            
        except Exception as e:
            return {"error": f"Erro na análise temporal: {str(e)}"}
    
    def analyze_temporal_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisar relacionamentos entre variáveis temporais e outras variáveis
        
        Args:
            df: DataFrame
            
        Returns:
            Análise de relacionamentos temporais
        """
        try:
            temporal_cols = []
            
            # Detectar colunas temporais
            for col in df.columns:
                if self._is_likely_temporal(df[col]):
                    temporal_cols.append(col)
            
            if not temporal_cols:
                return {"message": "Nenhuma coluna temporal detectada"}
            
            relationships = {}
            
            for temp_col in temporal_cols:
                # Converter para datetime
                try:
                    df_temp = df.copy()
                    df_temp[temp_col] = pd.to_datetime(df_temp[temp_col], errors='coerce')
                    df_temp = df_temp.dropna(subset=[temp_col])
                    
                    if len(df_temp) < 3:
                        continue
                    
                    # Analisar relacionamento com variáveis numéricas
                    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
                    
                    for num_col in numeric_cols:
                        correlation = self._calculate_temporal_correlation(
                            df_temp, temp_col, num_col
                        )
                        
                        if abs(correlation.get("correlation", 0)) > 0.3:
                            relationships[f"{temp_col}_vs_{num_col}"] = correlation
                            
                except Exception:
                    continue
            
            return {
                "temporal_relationships": relationships,
                "summary": {
                    "temporal_columns_found": len(temporal_cols),
                    "significant_relationships": len(relationships)
                }
            }
            
        except Exception as e:
            return {"error": f"Erro na análise de relacionamentos temporais: {str(e)}"}
    
    def _analyze_column_for_temporal_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analisar uma coluna para padrões temporais"""
        
        # Verificar se já é datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return {
                "is_temporal": True,
                "type": "datetime",
                "confidence": 1.0,
                "pattern": "datetime_type"
            }
        
        # Converter para string e analisar padrões
        str_series = series.astype(str).dropna()
        
        if len(str_series) == 0:
            return {"is_temporal": False}
        
        # Testar padrões de data
        pattern_matches = {}
        for i, pattern in enumerate(self.datetime_patterns):
            matches = str_series.str.match(pattern).sum()
            pattern_matches[f"pattern_{i}"] = matches / len(str_series)
        
        # Verificar maior taxa de correspondência
        best_match = max(pattern_matches.values())
        
        if best_match > 0.7:  # 70% das strings correspondem a um padrão
            return {
                "is_temporal": True,
                "type": "string_date",
                "confidence": best_match,
                "pattern": max(pattern_matches.items(), key=lambda x: x[1])[0]
            }
        
        # Verificar palavras-chave temporais
        temporal_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 
                           'year', 'month', 'day', 'hora', 'data', 'tempo']
        
        column_name = series.name.lower() if series.name else ""
        keyword_match = any(keyword in column_name for keyword in temporal_keywords)
        
        if keyword_match and best_match > 0.3:
            return {
                "is_temporal": True,
                "type": "keyword_match",
                "confidence": 0.8,
                "pattern": "keyword_based"
            }
        
        return {"is_temporal": False, "confidence": best_match}
    
    def _detect_frequency(self, date_series: pd.Series) -> str:
        """Detectar frequência da série temporal"""
        try:
            if len(date_series) < 2:
                return "irregular"
            
            # Calcular diferenças entre datas consecutivas
            diffs = date_series.diff().dropna()
            
            if len(diffs) == 0:
                return "irregular"
            
            # Encontrar o modo das diferenças
            most_common_diff = diffs.mode()
            
            if len(most_common_diff) == 0:
                return "irregular"
            
            diff_days = most_common_diff.iloc[0].days
            
            if diff_days == 1:
                return "daily"
            elif diff_days == 7:
                return "weekly"
            elif 28 <= diff_days <= 31:
                return "monthly"
            elif 365 <= diff_days <= 366:
                return "yearly"
            else:
                return f"custom_{diff_days}_days"
                
        except Exception:
            return "irregular"
    
    def _analyze_trend(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Analisar tendência da série temporal"""
        try:
            # Converter datas para números ordinais para regressão
            x = df[date_col].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = df[value_col].values
            
            # Regressão linear
            lr = LinearRegression()
            lr.fit(x, y)
            
            slope = lr.coef_[0]
            r_squared = lr.score(x, y)
            
            # Classificar tendência
            if abs(slope) < 1e-6:
                trend_type = "estável"
            elif slope > 0:
                trend_type = "crescente"
            else:
                trend_type = "decrescente"
            
            # Teste de significância da tendência
            y_pred = lr.predict(x)
            residuals = y - y_pred
            
            # Correlação de Pearson entre tempo e valores
            time_corr = stats.pearsonr(x.flatten(), y)[0] if len(x) > 2 else 0
            
            return {
                "trend_type": trend_type,
                "slope": float(slope),
                "r_squared": float(r_squared),
                "correlation_with_time": float(time_corr),
                "trend_strength": self._classify_trend_strength(r_squared),
                "is_significant": r_squared > 0.5 and abs(time_corr) > 0.7
            }
            
        except Exception as e:
            return {"error": f"Erro na análise de tendência: {str(e)}"}
    
    def _analyze_seasonality(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Analisar sazonalidade da série temporal"""
        try:
            df_temp = df.copy()
            
            # Extrair componentes temporais
            df_temp['month'] = df_temp[date_col].dt.month
            df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
            df_temp['quarter'] = df_temp[date_col].dt.quarter
            
            seasonality_results = {}
            
            # Análise mensal
            if df_temp['month'].nunique() > 1:
                monthly_stats = df_temp.groupby('month')[value_col].agg(['mean', 'std', 'count'])
                monthly_variation = monthly_stats['mean'].std() / monthly_stats['mean'].mean()
                seasonality_results['monthly'] = {
                    "variation_coefficient": float(monthly_variation),
                    "is_seasonal": monthly_variation > 0.2,
                    "peak_month": int(monthly_stats['mean'].idxmax()),
                    "low_month": int(monthly_stats['mean'].idxmin()),
                    "statistics": monthly_stats.to_dict()
                }
            
            # Análise por dia da semana
            if df_temp['day_of_week'].nunique() > 1:
                weekly_stats = df_temp.groupby('day_of_week')[value_col].agg(['mean', 'std', 'count'])
                weekly_variation = weekly_stats['mean'].std() / weekly_stats['mean'].mean()
                seasonality_results['weekly'] = {
                    "variation_coefficient": float(weekly_variation),
                    "is_seasonal": weekly_variation > 0.15,
                    "peak_day": int(weekly_stats['mean'].idxmax()),
                    "low_day": int(weekly_stats['mean'].idxmin()),
                    "statistics": weekly_stats.to_dict()
                }
            
            # Análise trimestral
            if df_temp['quarter'].nunique() > 1:
                quarterly_stats = df_temp.groupby('quarter')[value_col].agg(['mean', 'std', 'count'])
                quarterly_variation = quarterly_stats['mean'].std() / quarterly_stats['mean'].mean()
                seasonality_results['quarterly'] = {
                    "variation_coefficient": float(quarterly_variation),
                    "is_seasonal": quarterly_variation > 0.1,
                    "peak_quarter": int(quarterly_stats['mean'].idxmax()),
                    "low_quarter": int(quarterly_stats['mean'].idxmin()),
                    "statistics": quarterly_stats.to_dict()
                }
            
            return seasonality_results
            
        except Exception as e:
            return {"error": f"Erro na análise de sazonalidade: {str(e)}"}
    
    def _calculate_temporal_stats(self, df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
        """Calcular estatísticas temporais específicas"""
        try:
            values = df[value_col].dropna()
            
            if len(values) < 2:
                return {"error": "Dados insuficientes"}
            
            # Estatísticas básicas
            stats_dict = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min())
            }
            
            # Volatilidade (desvio padrão das diferenças)
            if len(values) > 1:
                differences = values.diff().dropna()
                stats_dict["volatility"] = float(differences.std()) if len(differences) > 0 else 0
            
            # Autocorrelação (lag-1)
            if len(values) > 2:
                autocorr = values.autocorr(lag=1)
                stats_dict["autocorrelation_lag1"] = float(autocorr) if not np.isnan(autocorr) else 0
            
            return stats_dict
            
        except Exception as e:
            return {"error": f"Erro no cálculo de estatísticas temporais: {str(e)}"}
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
        """Detectar anomalias temporais"""
        try:
            values = df[value_col].dropna()
            
            if len(values) < 4:
                return {"error": "Dados insuficientes para detecção de anomalias"}
            
            # Método IQR
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (values < lower_bound) | (values > upper_bound)
            outliers_count = outliers_mask.sum()
            
            # Método Z-score
            z_scores = np.abs(stats.zscore(values))
            z_outliers_mask = z_scores > 3
            z_outliers_count = z_outliers_mask.sum()
            
            return {
                "iqr_method": {
                    "outliers_count": int(outliers_count),
                    "outliers_percentage": float(outliers_count / len(values) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                },
                "zscore_method": {
                    "outliers_count": int(z_outliers_count),
                    "outliers_percentage": float(z_outliers_count / len(values) * 100),
                    "threshold": 3.0
                },
                "summary": {
                    "total_anomalies_iqr": int(outliers_count),
                    "total_anomalies_zscore": int(z_outliers_count),
                    "data_quality": "good" if outliers_count < len(values) * 0.05 else "requires_attention"
                }
            }
            
        except Exception as e:
            return {"error": f"Erro na detecção de anomalias: {str(e)}"}
    
    def _identify_temporal_patterns(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Identificar padrões temporais específicos"""
        try:
            patterns = {}
            
            # Padrão de crescimento/decrescimento consistente
            values = df[value_col].values
            if len(values) > 3:
                differences = np.diff(values)
                
                increasing_count = (differences > 0).sum()
                decreasing_count = (differences < 0).sum()
                
                if increasing_count / len(differences) > 0.8:
                    patterns["consistent_growth"] = True
                elif decreasing_count / len(differences) > 0.8:
                    patterns["consistent_decline"] = True
                else:
                    patterns["mixed_pattern"] = True
            
            # Padrão de estabilidade
            cv = df[value_col].std() / df[value_col].mean() if df[value_col].mean() != 0 else float('inf')
            patterns["stability"] = {
                "coefficient_of_variation": float(cv),
                "is_stable": cv < 0.1,
                "stability_level": "high" if cv < 0.1 else "medium" if cv < 0.3 else "low"
            }
            
            # Detecção de ciclos simples
            if len(values) >= 10:
                # Análise de autocorrelação para diferentes lags
                autocorrelations = []
                for lag in range(1, min(len(values)//3, 12)):
                    try:
                        autocorr = pd.Series(values).autocorr(lag=lag)
                        if not np.isnan(autocorr):
                            autocorrelations.append((lag, autocorr))
                    except:
                        continue
                
                # Encontrar lag com maior autocorrelação
                if autocorrelations:
                    best_lag = max(autocorrelations, key=lambda x: abs(x[1]))
                    if abs(best_lag[1]) > 0.5:
                        patterns["cyclical_pattern"] = {
                            "detected": True,
                            "cycle_length": best_lag[0],
                            "strength": abs(best_lag[1])
                        }
                    else:
                        patterns["cyclical_pattern"] = {"detected": False}
            
            return patterns
            
        except Exception as e:
            return {"error": f"Erro na identificação de padrões: {str(e)}"}
    
    def _is_likely_temporal(self, series: pd.Series) -> bool:
        """Verificar se uma coluna é provavelmente temporal"""
        analysis = self._analyze_column_for_temporal_patterns(series)
        return analysis.get("is_temporal", False)
    
    def _calculate_temporal_correlation(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Calcular correlação temporal entre variáveis"""
        try:
            # Converter data para ordinal para correlação
            date_ordinal = df[date_col].map(pd.Timestamp.toordinal)
            
            # Correlação entre tempo e valores
            correlation = date_ordinal.corr(df[value_col])
            
            return {
                "correlation": float(correlation) if not np.isnan(correlation) else 0,
                "relationship_type": "positive" if correlation > 0.3 else "negative" if correlation < -0.3 else "none",
                "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
            }
            
        except Exception:
            return {"correlation": 0, "relationship_type": "none", "strength": "none"}
    
    def _classify_trend_strength(self, r_squared: float) -> str:
        """Classificar força da tendência"""
        if r_squared > 0.8:
            return "muito_forte"
        elif r_squared > 0.6:
            return "forte"
        elif r_squared > 0.4:
            return "moderada"
        elif r_squared > 0.2:
            return "fraca"
        else:
            return "muito_fraca"
    
    def _generate_temporal_recommendations(self, temporal_columns: Dict) -> List[str]:
        """Gerar recomendações para análise temporal"""
        recommendations = []
        
        if len(temporal_columns) == 0:
            recommendations.append("Nenhuma coluna temporal detectada automaticamente")
        else:
            recommendations.append(f"Detectadas {len(temporal_columns)} colunas temporais")
            
            high_confidence = [col for col, info in temporal_columns.items() 
                             if info.get("confidence", 0) > 0.8]
            
            if high_confidence:
                recommendations.append(f"Colunas com alta confiança temporal: {', '.join(high_confidence[:3])}")
        
        return recommendations
    
    def _generate_time_series_recommendations(self, trend: Dict, seasonality: Dict, anomalies: Dict) -> List[str]:
        """Gerar recomendações para série temporal"""
        recommendations = []
        
        # Recomendações de tendência
        if trend.get("is_significant", False):
            trend_type = trend.get("trend_type", "")
            recommendations.append(f"Tendência {trend_type} significativa detectada")
        
        # Recomendações de sazonalidade
        seasonal_patterns = [k for k, v in seasonality.items() 
                           if isinstance(v, dict) and v.get("is_seasonal", False)]
        
        if seasonal_patterns:
            recommendations.append(f"Padrões sazonais detectados: {', '.join(seasonal_patterns)}")
        
        # Recomendações de anomalias
        if anomalies.get("summary", {}).get("data_quality") == "requires_attention":
            recommendations.append("Alto número de anomalias detectadas - verificar qualidade dos dados")
        
        return recommendations

# Instância global do serviço
temporal_analysis_service = TemporalAnalysisService()