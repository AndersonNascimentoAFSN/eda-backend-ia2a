"""
Script de teste para a funcionalidade completa de análise de dados
"""
import time
import requests
from pathlib import Path

# URL base da API
BASE_URL = "http://localhost:8000/api/v1"

def test_complete_analysis_flow():
    """Testa o fluxo completo: upload + análise de dados"""
    
    print("🧪 Testando fluxo completo de análise de dados\n")
    
    # 1. Verificar configuração do R2
    print("1️⃣ Verificando configuração do R2...")
    try:
        response = requests.get(f"{BASE_URL}/r2/config")
        config_data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Configurado: {config_data.get('configured', False)}")
        print(f"   Bucket: {config_data.get('bucket_name', 'N/A')}")
        
        if not config_data.get('configured', False):
            print("❌ R2 não está configurado. Configure as variáveis de ambiente.")
            return
            
    except Exception as e:
        print(f"❌ Erro ao verificar configuração: {e}")
        return
    
    # 2. Upload de arquivo de teste mais complexo
    print("\n2️⃣ Fazendo upload de arquivo de teste...")
    try:
        filename = "sales_data.csv"
        response = requests.post(
            f"{BASE_URL}/r2/presigned-upload",
            params={
                "filename": filename,
                "content_type": "text/csv",
                "folder": "analysis-test"
            }
        )
        
        if response.status_code == 200:
            upload_data = response.json()
            print(f"   ✅ URL gerada com sucesso!")
            print(f"   File Key: {upload_data['file_key']}")
            
            # Criar dados de teste mais complexos
            test_data = """date,product,category,sales,price,quantity,region,customer_age
2024-01-01,Laptop,Electronics,1500.00,1500.00,1,North,25
2024-01-02,Mouse,Electronics,25.99,25.99,1,South,30
2024-01-03,Keyboard,Electronics,79.99,79.99,1,East,35
2024-01-04,Monitor,Electronics,299.99,299.99,1,West,40
2024-01-05,Laptop,Electronics,1500.00,1500.00,1,North,28
2024-01-06,Tablet,Electronics,599.99,599.99,1,South,22
2024-01-07,Phone,Electronics,899.99,899.99,1,East,33
2024-01-08,Headphones,Electronics,199.99,199.99,1,West,29
2024-01-09,Speaker,Electronics,149.99,149.99,1,North,45
2024-01-10,Cable,Electronics,19.99,19.99,2,South,31
2024-01-11,Laptop,Electronics,1500.00,1500.00,1,East,26
2024-01-12,Mouse,Electronics,25.99,25.99,3,West,38
2024-01-13,Keyboard,Electronics,79.99,79.99,1,North,42
2024-01-14,Monitor,Electronics,299.99,299.99,2,South,27
2024-01-15,Laptop,Electronics,1500.00,1500.00,1,East,36"""
            
            # Upload para R2
            upload_response = requests.put(
                upload_data['upload_url'],
                data=test_data,
                headers={'Content-Type': upload_data['content_type']}
            )
            
            if upload_response.status_code in [200, 204]:
                print(f"   ✅ Arquivo enviado com sucesso!")
                file_key = upload_data['file_key']
            else:
                print(f"   ❌ Erro no upload: {upload_response.status_code}")
                return
                
        else:
            print(f"❌ Erro ao gerar URL: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Erro no upload: {e}")
        return
    
    # 3. Verificar tipos de análise disponíveis
    print("\n3️⃣ Verificando tipos de análise...")
    try:
        response = requests.get(f"{BASE_URL}/analysis/types")
        if response.status_code == 200:
            types_data = response.json()
            print(f"   ✅ Tipos disponíveis:")
            for analysis_type in types_data["available_types"]:
                print(f"     📊 {analysis_type['name']} ({analysis_type['id']})")
                print(f"         {analysis_type['description']}")
                print(f"         Duração: {analysis_type['estimated_duration']}")
        else:
            print(f"   ⚠️  Erro ao buscar tipos: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    # 4. Iniciar análise
    print("\n4️⃣ Iniciando análise de dados...")
    try:
        response = requests.post(
            f"{BASE_URL}/analysis/start",
            json={
                "file_key": file_key,
                "analysis_type": "basic_eda",
                "options": {"include_correlations": True}
            }
        )
        
        if response.status_code == 200:
            start_data = response.json()
            analysis_id = start_data["analysis_id"]
            print(f"   ✅ Análise iniciada!")
            print(f"   Analysis ID: {analysis_id}")
            print(f"   Status: {start_data['status']}")
            print(f"   Duração estimada: {start_data.get('estimated_duration_minutes', 'N/A')} minutos")
        else:
            print(f"   ❌ Erro ao iniciar análise: {response.status_code}")
            print(f"   Resposta: {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return
    
    # 5. Acompanhar progresso
    print("\n5️⃣ Acompanhando progresso da análise...")
    max_attempts = 30  # 30 tentativas = ~1 minuto
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/analysis/status/{analysis_id}")
            
            if response.status_code == 200:
                status_data = response.json()
                status = status_data["status"]
                progress = status_data["progress"]
                message = status_data["message"]
                
                print(f"   📊 Status: {status} | Progresso: {progress:.1f}% | {message}")
                
                if status == "completed":
                    print(f"   ✅ Análise concluída!")
                    break
                elif status == "error":
                    print(f"   ❌ Erro na análise: {status_data.get('error', 'Erro desconhecido')}")
                    return
                    
            else:
                print(f"   ⚠️  Erro ao verificar status: {response.status_code}")
            
            attempt += 1
            time.sleep(2)  # Aguardar 2 segundos
            
        except Exception as e:
            print(f"   ❌ Erro ao verificar status: {e}")
            attempt += 1
            time.sleep(2)
    
    if attempt >= max_attempts:
        print("   ⏰ Timeout - análise demorou mais que o esperado")
        return
    
    # 6. Obter resultados
    print("\n6️⃣ Obtendo resultados da análise...")
    try:
        response = requests.get(f"{BASE_URL}/analysis/results/{analysis_id}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Resultados obtidos com sucesso!")
            
            # Exibir resumo dos resultados
            dataset_info = results["dataset_info"]
            print(f"\n   📊 Resumo do Dataset:")
            print(f"      Arquivo: {dataset_info['filename']}")
            print(f"      Dimensões: {dataset_info['rows']} linhas × {dataset_info['columns']} colunas")
            print(f"      Tamanho: {dataset_info['memory_usage']:.2f} MB")
            
            print(f"\n   📈 Estatísticas:")
            summary = results["summary"]
            print(f"      Completude: {summary['completeness_score']}%")
            print(f"      Colunas numéricas: {summary.get('numeric_columns', 0)}")
            print(f"      Colunas categóricas: {summary.get('categorical_columns', 0)}")
            
            print(f"\n   📋 Colunas ({len(results['column_stats'])}):")
            for col in results['column_stats'][:5]:  # Mostrar apenas as primeiras 5
                print(f"      • {col['name']} ({col['dtype']}) - {col['non_null_count']}/{col['count']} valores")
                if col.get('mean') is not None:
                    print(f"        Média: {col['mean']:.2f}")
                if col.get('most_frequent'):
                    print(f"        Mais frequente: {col['most_frequent']}")
            
            if len(results['column_stats']) > 5:
                print(f"      ... e mais {len(results['column_stats']) - 5} colunas")
            
            print(f"\n   💡 Recomendações:")
            for rec in summary['recommendations']:
                print(f"      • {rec}")
            
        else:
            print(f"   ❌ Erro ao obter resultados: {response.status_code}")
            print(f"   Resposta: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    # 7. Limpar análise (opcional)
    print("\n7️⃣ Limpando análise...")
    try:
        response = requests.delete(f"{BASE_URL}/analysis/{analysis_id}")
        
        if response.status_code == 200:
            print(f"   ✅ Análise removida do cache")
        else:
            print(f"   ⚠️  Aviso: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ⚠️  Erro ao limpar: {e}")

    print(f"\n🎉 Teste do fluxo completo finalizado!")

if __name__ == "__main__":
    test_complete_analysis_flow()