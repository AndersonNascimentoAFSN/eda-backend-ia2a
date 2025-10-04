"""
Serviço de testes estatísticos avançados
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu, ttest_ind
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class StatisticalTestsService:
    """Serviço para testes estatísticos avançados"""
    
    def __init__(self):
        self.significance_level = 0.05
    
    def comprehensive_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executar bateria completa de testes estatísticos
        
        Args:
            df: DataFrame para análise
            
        Returns:
            Resultados de todos os testes estatísticos
        """
        try:
            results = {
                "normality_tests": self._test_normality_all_variables(df),
                "independence_tests": self._test_independence(df),
                "group_comparison_tests": self._test_group_comparisons(df),
                "correlation_significance": self._test_correlation_significance(df),
                "homogeneity_tests": self._test_homogeneity(df),
                "summary": {}
            }
            
            # Gerar resumo
            results["summary"] = self._generate_tests_summary(results)
            
            return results
            
        except Exception as e:
            return {"error": f"Erro nos testes estatísticos: {str(e)}"}
    
    def _test_normality_all_variables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Testar normalidade para todas as variáveis numéricas"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            normality_results = {}
            
            for col in numeric_cols:
                data = df[col].dropna()
                
                if len(data) < 3:
                    normality_results[col] = {"error": "Dados insuficientes"}
                    continue
                
                results = {}
                
                # Teste Shapiro-Wilk (melhor para n < 5000)
                if len(data) <= 5000:
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(data)
                        results["shapiro_wilk"] = {
                            "statistic": float(shapiro_stat),
                            "p_value": float(shapiro_p),
                            "is_normal": shapiro_p > self.significance_level,
                            "interpretation": "Normal" if shapiro_p > self.significance_level else "Não normal"
                        }
                    except Exception:
                        results["shapiro_wilk"] = {"error": "Falha no teste"}
                
                # Teste Kolmogorov-Smirnov
                try:
                    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                    results["kolmogorov_smirnov"] = {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": ks_p > self.significance_level,
                        "interpretation": "Normal" if ks_p > self.significance_level else "Não normal"
                    }
                except Exception:
                    results["kolmogorov_smirnov"] = {"error": "Falha no teste"}
                
                # Teste Anderson-Darling
                try:
                    ad_result = stats.anderson(data, dist='norm')
                    # Verificar se é normal no nível de 5%
                    critical_value_5pct = ad_result.critical_values[2]  # 5% level
                    is_normal_ad = ad_result.statistic < critical_value_5pct
                    
                    results["anderson_darling"] = {
                        "statistic": float(ad_result.statistic),
                        "critical_values": ad_result.critical_values.tolist(),
                        "significance_levels": ad_result.significance_level.tolist(),
                        "is_normal": is_normal_ad,
                        "interpretation": "Normal" if is_normal_ad else "Não normal"
                    }
                except Exception:
                    results["anderson_darling"] = {"error": "Falha no teste"}
                
                # Teste D'Agostino e Pearson
                try:
                    if len(data) >= 8:  # Requisito mínimo
                        dag_stat, dag_p = stats.normaltest(data)
                        results["dagostino_pearson"] = {
                            "statistic": float(dag_stat),
                            "p_value": float(dag_p),
                            "is_normal": dag_p > self.significance_level,
                            "interpretation": "Normal" if dag_p > self.significance_level else "Não normal"
                        }
                except Exception:
                    results["dagostino_pearson"] = {"error": "Falha no teste"}
                
                # Conclusão geral
                normal_tests = [test.get("is_normal", False) for test in results.values() 
                              if isinstance(test, dict) and "is_normal" in test]
                
                if normal_tests:
                    consensus = sum(normal_tests) / len(normal_tests)
                    results["consensus"] = {
                        "probably_normal": consensus >= 0.5,
                        "confidence": consensus,
                        "recommendation": "Assumir normalidade" if consensus >= 0.7 else 
                                       "Normalidade duvidosa" if consensus >= 0.3 else "Não assumir normalidade"
                    }
                
                normality_results[col] = results
            
            return normality_results
            
        except Exception as e:
            return {"error": f"Erro nos testes de normalidade: {str(e)}"}
    
    def _test_independence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Testar independência entre variáveis categóricas"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) < 2:
                return {"message": "Necessário pelo menos 2 variáveis categóricas"}
            
            independence_results = {}
            
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        # Criar tabela de contingência
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        
                        # Verificar se há células com frequência muito baixa
                        min_frequency = contingency_table.min().min()
                        total_cells = contingency_table.size
                        low_freq_cells = (contingency_table < 5).sum().sum()
                        
                        # Teste qui-quadrado
                        chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramér's V
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                        
                        # Teste exato de Fisher (para tabelas 2x2)
                        fisher_result = None
                        if contingency_table.shape == (2, 2) and min_frequency < 5:
                            try:
                                fisher_result = stats.fisher_exact(contingency_table)
                            except Exception:
                                fisher_result = None
                        
                        independence_results[f"{col1}_vs_{col2}"] = {
                            "chi_square_test": {
                                "statistic": float(chi2_stat),
                                "p_value": float(chi2_p),
                                "degrees_of_freedom": int(dof),
                                "is_independent": chi2_p > self.significance_level,
                                "interpretation": "Independentes" if chi2_p > self.significance_level else "Dependentes"
                            },
                            "cramers_v": {
                                "value": float(cramers_v),
                                "interpretation": self._interpret_cramers_v(cramers_v)
                            },
                            "data_quality": {
                                "min_frequency": int(min_frequency),
                                "low_frequency_cells": int(low_freq_cells),
                                "low_freq_percentage": float(low_freq_cells / total_cells * 100),
                                "test_validity": "válido" if low_freq_cells / total_cells < 0.2 else "questionável"
                            }
                        }
                        
                        # Adicionar resultado do teste de Fisher se aplicável
                        if fisher_result:
                            independence_results[f"{col1}_vs_{col2}"]["fisher_exact_test"] = {
                                "odds_ratio": float(fisher_result[0]),
                                "p_value": float(fisher_result[1]),
                                "is_independent": fisher_result[1] > self.significance_level,
                                "interpretation": "Independentes" if fisher_result[1] > self.significance_level else "Dependentes"
                            }
                            
                    except Exception as e:
                        independence_results[f"{col1}_vs_{col2}"] = {"error": str(e)}
            
            return independence_results
            
        except Exception as e:
            return {"error": f"Erro nos testes de independência: {str(e)}"}
    
    def _test_group_comparisons(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Testar diferenças entre grupos"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not numeric_cols or not categorical_cols:
                return {"message": "Necessário variáveis numéricas e categóricas"}
            
            group_comparison_results = {}
            
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    try:
                        # Filtrar dados válidos
                        clean_data = df[[num_col, cat_col]].dropna()
                        
                        if len(clean_data) < 3:
                            continue
                        
                        # Obter grupos
                        groups = [group[num_col].values for name, group in clean_data.groupby(cat_col) 
                                if len(group) >= 2]
                        
                        if len(groups) < 2:
                            continue
                        
                        results = {}
                        
                        # ANOVA (se mais de 2 grupos)
                        if len(groups) > 2:
                            try:
                                f_stat, f_p = f_oneway(*groups)
                                results["anova"] = {
                                    "f_statistic": float(f_stat),
                                    "p_value": float(f_p),
                                    "significant_difference": f_p < self.significance_level,
                                    "interpretation": "Diferenças significativas entre grupos" if f_p < self.significance_level 
                                                    else "Sem diferenças significativas"
                                }
                            except Exception:
                                results["anova"] = {"error": "Falha no teste ANOVA"}
                            
                            # Kruskal-Wallis (alternativa não-paramétrica)
                            try:
                                h_stat, h_p = kruskal(*groups)
                                results["kruskal_wallis"] = {
                                    "h_statistic": float(h_stat),
                                    "p_value": float(h_p),
                                    "significant_difference": h_p < self.significance_level,
                                    "interpretation": "Diferenças significativas entre grupos" if h_p < self.significance_level 
                                                    else "Sem diferenças significativas"
                                }
                            except Exception:
                                results["kruskal_wallis"] = {"error": "Falha no teste Kruskal-Wallis"}
                        
                        # Para 2 grupos
                        elif len(groups) == 2:
                            # Teste t de Student
                            try:
                                t_stat, t_p = ttest_ind(groups[0], groups[1])
                                results["t_test"] = {
                                    "t_statistic": float(t_stat),
                                    "p_value": float(t_p),
                                    "significant_difference": t_p < self.significance_level,
                                    "interpretation": "Diferenças significativas entre grupos" if t_p < self.significance_level 
                                                    else "Sem diferenças significativas"
                                }
                            except Exception:
                                results["t_test"] = {"error": "Falha no teste t"}
                            
                            # Mann-Whitney U (alternativa não-paramétrica)
                            try:
                                u_stat, u_p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                                results["mann_whitney"] = {
                                    "u_statistic": float(u_stat),
                                    "p_value": float(u_p),
                                    "significant_difference": u_p < self.significance_level,
                                    "interpretation": "Diferenças significativas entre grupos" if u_p < self.significance_level 
                                                    else "Sem diferenças significativas"
                                }
                            except Exception:
                                results["mann_whitney"] = {"error": "Falha no teste Mann-Whitney"}
                        
                        # Estatísticas descritivas por grupo
                        group_stats = {}
                        for name, group in clean_data.groupby(cat_col):
                            if len(group) >= 2:
                                group_stats[str(name)] = {
                                    "count": int(len(group)),
                                    "mean": float(group[num_col].mean()),
                                    "std": float(group[num_col].std()),
                                    "median": float(group[num_col].median())
                                }
                        
                        results["group_statistics"] = group_stats
                        
                        group_comparison_results[f"{num_col}_by_{cat_col}"] = results
                        
                    except Exception as e:
                        group_comparison_results[f"{num_col}_by_{cat_col}"] = {"error": str(e)}
            
            return group_comparison_results
            
        except Exception as e:
            return {"error": f"Erro nos testes de comparação de grupos: {str(e)}"}
    
    def _test_correlation_significance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Testar significância das correlações"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {"message": "Necessário pelo menos 2 variáveis numéricas"}
            
            correlation_tests = {}
            
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    try:
                        # Dados limpos
                        clean_data = df[[col1, col2]].dropna()
                        
                        if len(clean_data) < 3:
                            continue
                        
                        x, y = clean_data[col1], clean_data[col2]
                        
                        # Correlação de Pearson
                        pearson_r, pearson_p = stats.pearsonr(x, y)
                        
                        # Correlação de Spearman
                        spearman_r, spearman_p = stats.spearmanr(x, y)
                        
                        # Correlação de Kendall
                        kendall_tau, kendall_p = stats.kendalltau(x, y)
                        
                        correlation_tests[f"{col1}_vs_{col2}"] = {
                            "pearson": {
                                "correlation": float(pearson_r),
                                "p_value": float(pearson_p),
                                "is_significant": pearson_p < self.significance_level,
                                "strength": self._interpret_correlation_strength(abs(pearson_r)),
                                "direction": "positiva" if pearson_r > 0 else "negativa"
                            },
                            "spearman": {
                                "correlation": float(spearman_r),
                                "p_value": float(spearman_p),
                                "is_significant": spearman_p < self.significance_level,
                                "strength": self._interpret_correlation_strength(abs(spearman_r)),
                                "direction": "positiva" if spearman_r > 0 else "negativa"
                            },
                            "kendall": {
                                "correlation": float(kendall_tau),
                                "p_value": float(kendall_p),
                                "is_significant": kendall_p < self.significance_level,
                                "strength": self._interpret_correlation_strength(abs(kendall_tau)),
                                "direction": "positiva" if kendall_tau > 0 else "negativa"
                            }
                        }
                        
                    except Exception as e:
                        correlation_tests[f"{col1}_vs_{col2}"] = {"error": str(e)}
            
            return correlation_tests
            
        except Exception as e:
            return {"error": f"Erro nos testes de correlação: {str(e)}"}
    
    def _test_homogeneity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Testar homogeneidade de variâncias"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not numeric_cols or not categorical_cols:
                return {"message": "Necessário variáveis numéricas e categóricas"}
            
            homogeneity_results = {}
            
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    try:
                        # Filtrar dados válidos
                        clean_data = df[[num_col, cat_col]].dropna()
                        
                        if len(clean_data) < 6:  # Mínimo para teste de Levene
                            continue
                        
                        # Obter grupos
                        groups = [group[num_col].values for name, group in clean_data.groupby(cat_col) 
                                if len(group) >= 2]
                        
                        if len(groups) < 2:
                            continue
                        
                        results = {}
                        
                        # Teste de Levene
                        try:
                            levene_stat, levene_p = stats.levene(*groups)
                            results["levene"] = {
                                "statistic": float(levene_stat),
                                "p_value": float(levene_p),
                                "homogeneous_variances": levene_p > self.significance_level,
                                "interpretation": "Variâncias homogêneas" if levene_p > self.significance_level 
                                                else "Variâncias heterogêneas"
                            }
                        except Exception:
                            results["levene"] = {"error": "Falha no teste de Levene"}
                        
                        # Teste de Bartlett
                        try:
                            bartlett_stat, bartlett_p = stats.bartlett(*groups)
                            results["bartlett"] = {
                                "statistic": float(bartlett_stat),
                                "p_value": float(bartlett_p),
                                "homogeneous_variances": bartlett_p > self.significance_level,
                                "interpretation": "Variâncias homogêneas" if bartlett_p > self.significance_level 
                                                else "Variâncias heterogêneas"
                            }
                        except Exception:
                            results["bartlett"] = {"error": "Falha no teste de Bartlett"}
                        
                        # Estatísticas de variância por grupo
                        variance_stats = {}
                        for name, group in clean_data.groupby(cat_col):
                            if len(group) >= 2:
                                variance_stats[str(name)] = {
                                    "variance": float(group[num_col].var()),
                                    "std": float(group[num_col].std()),
                                    "count": int(len(group))
                                }
                        
                        results["variance_statistics"] = variance_stats
                        
                        homogeneity_results[f"{num_col}_by_{cat_col}"] = results
                        
                    except Exception as e:
                        homogeneity_results[f"{num_col}_by_{cat_col}"] = {"error": str(e)}
            
            return homogeneity_results
            
        except Exception as e:
            return {"error": f"Erro nos testes de homogeneidade: {str(e)}"}
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpretar valor de Cramér's V"""
        if cramers_v < 0.1:
            return "Associação negligível"
        elif cramers_v < 0.3:
            return "Associação fraca"
        elif cramers_v < 0.5:
            return "Associação moderada"
        else:
            return "Associação forte"
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpretar força da correlação"""
        if correlation < 0.1:
            return "negligível"
        elif correlation < 0.3:
            return "fraca"
        elif correlation < 0.5:
            return "moderada"
        elif correlation < 0.7:
            return "forte"
        else:
            return "muito forte"
    
    def _generate_tests_summary(self, results: Dict) -> Dict[str, Any]:
        """Gerar resumo dos testes estatísticos"""
        summary = {
            "total_tests_performed": 0,
            "significant_results": 0,
            "normality_summary": {},
            "independence_summary": {},
            "group_differences_summary": {},
            "correlation_summary": {},
            "recommendations": []
        }
        
        try:
            # Resumo de normalidade
            normality = results.get("normality_tests", {})
            if normality and not normality.get("error"):
                normal_vars = [col for col, test in normality.items() 
                             if isinstance(test, dict) and test.get("consensus", {}).get("probably_normal", False)]
                
                summary["normality_summary"] = {
                    "variables_tested": len(normality),
                    "probably_normal": len(normal_vars),
                    "normal_variables": normal_vars
                }
                
                if len(normal_vars) > 0:
                    summary["recommendations"].append(f"Variáveis que podem assumir normalidade: {', '.join(normal_vars[:3])}")
            
            # Resumo de independência
            independence = results.get("independence_tests", {})
            if independence and not independence.get("error"):
                dependent_pairs = [pair for pair, test in independence.items() 
                                 if isinstance(test, dict) and not test.get("chi_square_test", {}).get("is_independent", True)]
                
                summary["independence_summary"] = {
                    "pairs_tested": len(independence),
                    "dependent_pairs": len(dependent_pairs),
                    "dependent_relationships": dependent_pairs
                }
                
                if dependent_pairs:
                    summary["recommendations"].append(f"Relações de dependência encontradas em {len(dependent_pairs)} pares de variáveis")
            
            # Resumo de comparações de grupo
            group_tests = results.get("group_comparison_tests", {})
            if group_tests and not group_tests.get("error"):
                significant_differences = []
                for test_name, test_result in group_tests.items():
                    if isinstance(test_result, dict):
                        for method, method_result in test_result.items():
                            if isinstance(method_result, dict) and method_result.get("significant_difference", False):
                                significant_differences.append(test_name)
                                break
                
                summary["group_differences_summary"] = {
                    "comparisons_tested": len(group_tests),
                    "significant_differences": len(set(significant_differences)),
                    "significant_comparisons": list(set(significant_differences))
                }
            
            # Resumo de correlações
            correlations = results.get("correlation_significance", {})
            if correlations and not correlations.get("error"):
                significant_corrs = [pair for pair, test in correlations.items() 
                                   if isinstance(test, dict) and test.get("pearson", {}).get("is_significant", False)]
                
                summary["correlation_summary"] = {
                    "pairs_tested": len(correlations),
                    "significant_correlations": len(significant_corrs),
                    "significant_pairs": significant_corrs
                }
            
            # Contar total de testes
            summary["total_tests_performed"] = (
                len(normality) * 4 +  # 4 testes de normalidade por variável
                len(independence) +   # 1 teste por par de variáveis categóricas
                len(group_tests) * 2 + # 2 testes por comparação de grupo
                len(correlations) * 3  # 3 testes de correlação por par
            )
            
            return summary
            
        except Exception:
            return summary

# Instância global do serviço
statistical_tests_service = StatisticalTestsService()