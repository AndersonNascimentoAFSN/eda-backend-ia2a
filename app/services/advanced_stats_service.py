"""
Serviço de análise estatística avançada
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedStatsService:
    """Serviço para análises estatísticas avançadas"""
    
    def __init__(self):
        self.random_state = 42
    
    def analyze_distributions(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Análise detalhada de distribuição de uma variável numérica
        
        Args:
            df: DataFrame
            column: Nome da coluna
            
        Returns:
            Análise de distribuição
        """
        try:
            data = df[column].dropna()
            
            if len(data) < 3:
                return {"error": "Dados insuficientes para análise"}
            
            # Testes de normalidade
            shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limitar para performance
            
            # Teste Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            
            # Análise de assimetria e curtose
            skewness = stats.skew(data)
            kurtosis_val = stats.kurtosis(data)
            
            # Detecção do tipo de distribuição
            distribution_type = self._detect_distribution_type(skewness, kurtosis_val, shapiro_p)
            
            return {
                "normality_tests": {
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05
                    },
                    "kolmogorov_smirnov": {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": ks_p > 0.05
                    }
                },
                "distribution_characteristics": {
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis_val),
                    "distribution_type": distribution_type
                },
                "interpretation": {
                    "skewness_interpretation": self._interpret_skewness(skewness),
                    "kurtosis_interpretation": self._interpret_kurtosis(kurtosis_val),
                    "normality_conclusion": "Normal" if (shapiro_p > 0.05 and ks_p > 0.05) else "Não normal"
                }
            }
            
        except Exception as e:
            return {"error": f"Erro na análise de distribuição: {str(e)}"}
    
    def generate_cross_tables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gerar tabelas cruzadas entre variáveis categóricas
        
        Args:
            df: DataFrame
            
        Returns:
            Análise de tabelas cruzadas
        """
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) < 2:
                return {"message": "Necessário pelo menos 2 variáveis categóricas"}
            
            cross_tables = {}
            significant_associations = []
            
            # Analisar pares de variáveis categóricas
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        # Criar tabela cruzada
                        cross_tab = pd.crosstab(df[col1], df[col2])
                        
                        # Teste qui-quadrado
                        chi2, p_value, dof, expected = stats.chi2_contingency(cross_tab)
                        
                        # Cramér's V (medida de associação)
                        n = cross_tab.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(cross_tab.shape) - 1)))
                        
                        cross_tables[f"{col1}_vs_{col2}"] = {
                            "cross_table": cross_tab.to_dict(),
                            "chi_square_test": {
                                "chi2_statistic": float(chi2),
                                "p_value": float(p_value),
                                "degrees_of_freedom": int(dof),
                                "is_significant": p_value < 0.05
                            },
                            "cramers_v": float(cramers_v),
                            "association_strength": self._interpret_cramers_v(cramers_v)
                        }
                        
                        if p_value < 0.05:
                            significant_associations.append({
                                "variables": [col1, col2],
                                "p_value": float(p_value),
                                "cramers_v": float(cramers_v),
                                "strength": self._interpret_cramers_v(cramers_v)
                            })
                            
                    except Exception as e:
                        cross_tables[f"{col1}_vs_{col2}"] = {"error": str(e)}
            
            return {
                "cross_tables": cross_tables,
                "significant_associations": significant_associations,
                "summary": {
                    "total_pairs_analyzed": len(cross_tables),
                    "significant_associations_count": len(significant_associations)
                }
            }
            
        except Exception as e:
            return {"error": f"Erro na análise de tabelas cruzadas: {str(e)}"}
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Análise de importância de features
        
        Args:
            df: DataFrame
            target_column: Coluna alvo (opcional)
            
        Returns:
            Análise de importância das variáveis
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {"message": "Necessário pelo menos 2 variáveis numéricas"}
            
            results = {}
            
            # Se não especificou target, usar a primeira variável numérica
            if target_column is None:
                target_column = numeric_cols[0]
            
            if target_column not in numeric_cols:
                return {"error": f"Coluna alvo '{target_column}' deve ser numérica"}
            
            feature_cols = [col for col in numeric_cols if col != target_column]
            
            if len(feature_cols) == 0:
                return {"error": "Nenhuma feature disponível"}
            
            # Preparar dados
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df[target_column].fillna(df[target_column].median())
            
            # Random Forest Feature Importance
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            
            rf_importance = dict(zip(feature_cols, rf.feature_importances_))
            
            # Mutual Information
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            mi_importance = dict(zip(feature_cols, mi_scores))
            
            # Correlation com target
            correlations = {}
            for col in feature_cols:
                corr = df[col].corr(df[target_column])
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
            
            # Multicolinearidade (VIF)
            vif_scores = self._calculate_vif(X)
            
            # Ranking combinado
            combined_ranking = self._combine_importance_scores(
                rf_importance, mi_importance, correlations
            )
            
            results = {
                "target_variable": target_column,
                "feature_importance": {
                    "random_forest": dict(sorted(rf_importance.items(), 
                                                key=lambda x: x[1], reverse=True)),
                    "mutual_information": dict(sorted(mi_importance.items(), 
                                                     key=lambda x: x[1], reverse=True)),
                    "correlation_with_target": dict(sorted(correlations.items(), 
                                                          key=lambda x: x[1], reverse=True))
                },
                "multicollinearity": vif_scores,
                "combined_ranking": combined_ranking,
                "recommendations": self._generate_feature_recommendations(
                    combined_ranking, vif_scores
                )
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Erro na análise de importância: {str(e)}"}
    
    def perform_clustering_analysis(self, df: pd.DataFrame, max_clusters: int = 5) -> Dict[str, Any]:
        """
        Análise de clustering
        
        Args:
            df: DataFrame
            max_clusters: Número máximo de clusters
            
        Returns:
            Análise de clustering
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {"message": "Necessário pelo menos 2 variáveis numéricas"}
            
            # Preparar dados
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Padronizar dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means - encontrar número ótimo de clusters
            inertias = []
            silhouette_scores = []
            
            for k in range(2, min(max_clusters + 1, len(df) // 2)):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                inertias.append(kmeans.inertia_)
                
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
            
            # Melhor número de clusters (maior silhouette score)
            best_k = np.argmax(silhouette_scores) + 2
            
            # Clustering final
            kmeans_final = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
            kmeans_labels = kmeans_final.fit_predict(X_scaled)
            
            # DBSCAN para detecção de outliers
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            # Estatísticas dos clusters
            cluster_stats = self._analyze_cluster_characteristics(
                df, numeric_cols, kmeans_labels, best_k
            )
            
            return {
                "kmeans_analysis": {
                    "optimal_clusters": int(best_k),
                    "silhouette_scores": dict(zip(range(2, len(silhouette_scores) + 2), silhouette_scores)),
                    "inertia_scores": dict(zip(range(2, len(inertias) + 2), inertias)),
                    "cluster_labels": kmeans_labels.tolist(),
                    "cluster_statistics": cluster_stats
                },
                "outlier_detection": {
                    "dbscan_labels": dbscan_labels.tolist(),
                    "outliers_count": np.sum(dbscan_labels == -1),
                    "outliers_percentage": float(np.sum(dbscan_labels == -1) / len(dbscan_labels) * 100)
                },
                "recommendations": self._generate_clustering_recommendations(
                    best_k, len(df), np.sum(dbscan_labels == -1)
                )
            }
            
        except Exception as e:
            return {"error": f"Erro na análise de clustering: {str(e)}"}
    
    def _detect_distribution_type(self, skewness: float, kurtosis: float, shapiro_p: float) -> str:
        """Detectar tipo de distribuição"""
        if shapiro_p > 0.05:
            return "Normal"
        elif abs(skewness) < 0.5:
            return "Aproximadamente simétrica"
        elif skewness > 0.5:
            return "Assimétrica positiva (cauda à direita)"
        elif skewness < -0.5:
            return "Assimétrica negativa (cauda à esquerda)"
        else:
            return "Indefinida"
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpretar valor de assimetria"""
        if abs(skewness) < 0.5:
            return "Aproximadamente simétrica"
        elif 0.5 <= abs(skewness) <= 1:
            return "Moderadamente assimétrica"
        else:
            return "Altamente assimétrica"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpretar valor de curtose"""
        if abs(kurtosis) < 0.5:
            return "Mesocúrtica (normal)"
        elif kurtosis > 0.5:
            return "Leptocúrtica (caudas pesadas)"
        else:
            return "Platicúrtica (caudas leves)"
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpretar Cramér's V"""
        if cramers_v < 0.1:
            return "Associação negligível"
        elif cramers_v < 0.3:
            return "Associação fraca"
        elif cramers_v < 0.5:
            return "Associação moderada"
        else:
            return "Associação forte"
    
    def _calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calcular Variance Inflation Factor"""
        try:
            vif_data = {}
            for i, col in enumerate(X.columns):
                vif_data[col] = variance_inflation_factor(X.values, i)
            return vif_data
        except:
            return {}
    
    def _combine_importance_scores(self, rf_imp: Dict, mi_imp: Dict, corr_imp: Dict) -> List[Dict]:
        """Combinar diferentes medidas de importância"""
        features = list(rf_imp.keys())
        combined = []
        
        for feature in features:
            # Normalizar scores para 0-1
            rf_score = rf_imp.get(feature, 0)
            mi_score = mi_imp.get(feature, 0)
            corr_score = corr_imp.get(feature, 0)
            
            # Score combinado (média ponderada)
            combined_score = (rf_score * 0.4 + mi_score * 0.3 + corr_score * 0.3)
            
            combined.append({
                "feature": feature,
                "combined_score": float(combined_score),
                "rf_importance": float(rf_score),
                "mutual_info": float(mi_score),
                "correlation": float(corr_score)
            })
        
        return sorted(combined, key=lambda x: x["combined_score"], reverse=True)
    
    def _generate_feature_recommendations(self, ranking: List[Dict], vif_scores: Dict) -> List[str]:
        """Gerar recomendações baseadas na importância das features"""
        recommendations = []
        
        if len(ranking) > 0:
            top_features = [item["feature"] for item in ranking[:3]]
            recommendations.append(f"Features mais importantes: {', '.join(top_features)}")
        
        # Verificar multicolinearidade
        high_vif = [col for col, vif in vif_scores.items() if vif > 10]
        if high_vif:
            recommendations.append(f"Alta multicolinearidade detectada em: {', '.join(high_vif[:3])}")
        
        return recommendations
    
    def _analyze_cluster_characteristics(self, df: pd.DataFrame, numeric_cols: List[str], 
                                       labels: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Analisar características dos clusters"""
        cluster_stats = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = df[numeric_cols][cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_stats[f"cluster_{cluster_id}"] = {
                    "size": int(len(cluster_data)),
                    "percentage": float(len(cluster_data) / len(df) * 100),
                    "means": cluster_data.mean().to_dict(),
                    "stds": cluster_data.std().to_dict()
                }
        
        return cluster_stats
    
    def _generate_clustering_recommendations(self, n_clusters: int, n_samples: int, 
                                           n_outliers: int) -> List[str]:
        """Gerar recomendações de clustering"""
        recommendations = []
        
        recommendations.append(f"Número ótimo de clusters identificado: {n_clusters}")
        
        if n_outliers > 0:
            outlier_pct = (n_outliers / n_samples) * 100
            recommendations.append(f"Detectados {n_outliers} outliers ({outlier_pct:.1f}% dos dados)")
        
        if n_clusters > 5:
            recommendations.append("Muitos clusters podem indicar dados muito heterogêneos")
        
        return recommendations

# Instância global do serviço
advanced_stats_service = AdvancedStatsService()