#!/usr/bin/env python3
"""
Teste da detecção automática de separador CSV
"""
import requests
import csv
import io
import tempfile
import os

def create_test_csv_semicolon():
    """Cria um CSV de teste com separador ponto e vírgula"""
    data = [
        ["id", "nome", "idade", "cidade", "salario"],
        ["1", "João Silva", "30", "São Paulo", "5000.50"],
        ["2", "Maria Santos", "25", "Rio de Janeiro", "4500.75"],
        ["3", "Pedro Costa", "35", "Belo Horizonte", "6000.00"],
        ["4", "Ana Oliveira", "28", "Porto Alegre", "5500.25"]
    ]
    
    # Criar arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(data)
        return f.name

def create_test_csv_comma():
    """Cria um CSV de teste com separador vírgula"""
    data = [
        ["id", "nome", "idade", "cidade", "salario"],
        ["1", "João Silva", "30", "São Paulo", "5000.50"],
        ["2", "Maria Santos", "25", "Rio de Janeiro", "4500.75"],
        ["3", "Pedro Costa", "35", "Belo Horizonte", "6000.00"],
        ["4", "Ana Oliveira", "28", "Porto Alegre", "5500.25"]
    ]
    
    # Criar arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)
        return f.name

def test_upload_and_analysis(file_path, test_name):
    """Testa upload e análise de um arquivo CSV"""
    print(f"\n🧪 Testando: {test_name}")
    print(f"📁 Arquivo: {file_path}")
    
    try:
        # 1. Obter URL pré-assinada
        print("1️⃣ Obtendo URL pré-assinada...")
        filename = os.path.basename(file_path)
        
        response = requests.post(
            "http://localhost:8000/api/v1/r2/presigned-upload/",
            params={"filename": filename, "folder": "test-uploads"}
        )
        
        if response.status_code != 200:
            print(f"❌ Erro ao obter URL pré-assinada: {response.status_code}")
            print(response.text)
            return False
        
        upload_data = response.json()
        print(f"✅ URL obtida: {upload_data['file_key']}")
        
        # 2. Upload do arquivo
        print("2️⃣ Fazendo upload...")
        with open(file_path, 'rb') as f:
            upload_response = requests.put(
                upload_data['upload_url'],
                data=f.read(),
                headers={'Content-Type': 'text/csv'}
            )
        
        if upload_response.status_code not in [200, 204]:
            print(f"❌ Erro no upload: {upload_response.status_code}")
            return False
        
        print("✅ Upload concluído")
        
        # 3. Iniciar análise
        print("3️⃣ Iniciando análise...")
        analysis_response = requests.post(
            "http://localhost:8000/api/v1/analysis/start",
            json={
                "file_key": upload_data['file_key'],
                "analysis_type": "basic_eda"
            }
        )
        
        if analysis_response.status_code != 200:
            print(f"❌ Erro ao iniciar análise: {analysis_response.status_code}")
            print(analysis_response.text)
            return False
        
        analysis_data = analysis_response.json()
        print(f"✅ Análise iniciada: {analysis_data['analysis_id']}")
        
        # 4. Verificar status (aguardar um pouco)
        import time
        time.sleep(5)
        
        print("4️⃣ Verificando status...")
        status_response = requests.get(
            f"http://localhost:8000/api/v1/analysis/status/{analysis_data['analysis_id']}"
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"📊 Status: {status_data['status']}")
            print(f"📈 Progresso: {status_data.get('progress', 0)}%")
            
            if status_data['status'] == 'completed':
                # 5. Obter resultados
                print("5️⃣ Obtendo resultados...")
                results_response = requests.get(
                    f"http://localhost:8000/api/v1/analysis/results/{analysis_data['analysis_id']}"
                )
                
                if results_response.status_code == 200:
                    results_data = results_response.json()
                    
                    # Verificar se detectou as colunas corretas
                    if 'dataset_info' in results_data.get('results', {}):
                        dataset_info = results_data['results']['dataset_info']
                        num_columns = dataset_info.get('columns', 0)
                        column_names = dataset_info.get('column_names', [])
                        
                        print(f"📋 Colunas detectadas: {num_columns}")
                        print(f"📝 Nomes das colunas: {column_names}")
                        
                        if num_columns >= 5:  # Esperamos 5 colunas
                            print("✅ Separador detectado corretamente!")
                            return True
                        else:
                            print("❌ Separador não detectado corretamente")
                            return False
                
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {str(e)}")
        return False
    
    finally:
        # Limpar arquivo temporário
        if os.path.exists(file_path):
            os.unlink(file_path)

def main():
    """Função principal de teste"""
    print("🔍 Teste de Detecção Automática de Separador CSV")
    print("=" * 50)
    
    # Teste 1: CSV com ponto e vírgula
    semicolon_file = create_test_csv_semicolon()
    success1 = test_upload_and_analysis(semicolon_file, "CSV com separador ';'")
    
    # Teste 2: CSV com vírgula
    comma_file = create_test_csv_comma()
    success2 = test_upload_and_analysis(comma_file, "CSV com separador ','")
    
    print("\n" + "=" * 50)
    print("📋 Resultados dos Testes:")
    print(f"   Separador ';': {'✅ Passou' if success1 else '❌ Falhou'}")
    print(f"   Separador ',': {'✅ Passou' if success2 else '❌ Falhou'}")
    
    if success1 and success2:
        print("\n🎉 Todos os testes passaram! Detecção de separador funcionando.")
    else:
        print("\n⚠️  Alguns testes falharam. Verifique a implementação.")

if __name__ == "__main__":
    main()