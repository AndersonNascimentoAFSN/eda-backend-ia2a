"""
Script de teste para funcionalidade de upload do Cloudflare R2
"""
import requests
from pathlib import Path

# URL base da API
BASE_URL = "http://localhost:8000/api/v1"

def test_r2_functionality():
    """Testa a funcionalidade completa do R2"""
    
    print("🧪 Testando funcionalidade do Cloudflare R2\n")
    
    # 1. Verificar configuração do R2
    print("1️⃣ Verificando configuração do R2...")
    try:
        response = requests.get(f"{BASE_URL}/r2/config")
        config_data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Configurado: {config_data.get('configured', False)}")
        print(f"   Bucket: {config_data.get('bucket_name', 'N/A')}")
        print(f"   Mensagem: {config_data.get('message', 'N/A')}")
        
        if not config_data.get('configured', False):
            print("❌ R2 não está configurado. Configure as variáveis de ambiente.")
            return
            
    except Exception as e:
        print(f"❌ Erro ao verificar configuração: {e}")
        return
    
    # 2. Gerar URL pré-assinada para upload
    print("\n2️⃣ Gerando URL pré-assinada para upload...")
    try:
        filename = "test_data.csv"
        response = requests.post(
            f"{BASE_URL}/r2/presigned-upload",
            params={
                "filename": filename,
                "content_type": "text/csv",
                "folder": "test-uploads"
            }
        )
        
        if response.status_code == 200:
            upload_data = response.json()
            print(f"   ✅ URL gerada com sucesso!")
            print(f"   File Key: {upload_data['file_key']}")
            print(f"   Expira em: {upload_data['expires_in']} segundos")
            print(f"   Tamanho máximo: {upload_data['max_file_size_mb']} MB")
            
            # Salvar dados para uso posterior
            file_key = upload_data['file_key']
            upload_url = upload_data['upload_url']
            
        else:
            print(f"❌ Erro ao gerar URL: {response.status_code}")
            print(f"   {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Erro ao gerar URL pré-assinada: {e}")
        return
    
    # 3. Upload real simplificado
    print(f"\n3️⃣ Fazendo upload real...")
    
    # Preparar dados do arquivo
    test_data = "name,age,city\nJohn,30,New York\nJane,25,San Francisco"
    
    # Headers simples - apenas Content-Type
    headers = {
        'Content-Type': upload_data['content_type']
    }
    
    try:
        response = requests.put(
            upload_data['upload_url'],
            data=test_data,
            headers=headers
        )
        
        if response.status_code in [200, 204]:
            print(f"   ✅ Upload bem-sucedido! Status: {response.status_code}")
            print(f"   📁 Arquivo salvo como: {file_key}")
            file_uploaded = True
        else:
            print(f"   ❌ Erro no upload: {response.status_code}")
            print(f"   Resposta: {response.text}")
            file_uploaded = False
            return
            
    except Exception as e:
        print(f"   ❌ Exceção no upload: {e}")
        return
    
    # 4. Verificar se o arquivo foi uploaded
    if file_uploaded:
        print(f"\n4️⃣ Verificando informações do arquivo...")
        try:
            response = requests.get(
                f"{BASE_URL}/r2/file-info",
                params={"file_key": file_key}
            )
            
            if response.status_code == 200:
                file_info = response.json()
                print(f"   ✅ Arquivo encontrado!")
                print(f"   Tamanho: {file_info['size']} bytes")
                print(f"   Última modificação: {file_info['last_modified']}")
                print(f"   ETag: {file_info['etag']}")
            else:
                print(f"   ⚠️  Arquivo não encontrado: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Erro ao verificar arquivo: {e}")
    
    # 5. Gerar URL de download
    if file_uploaded:
        print(f"\n5️⃣ Gerando URL de download...")
        try:
            response = requests.post(
                f"{BASE_URL}/r2/presigned-download",
                params={"file_key": file_key}
            )
            
            if response.status_code == 200:
                download_data = response.json()
                print(f"   ✅ URL de download gerada!")
                print(f"   URL expira em: {download_data['expires_in']} segundos")
                
                # Testar download
                download_response = requests.get(download_data['download_url'])
                if download_response.status_code == 200:
                    print(f"   ✅ Download testado com sucesso!")
                    print(f"   Conteúdo: {download_response.text[:50]}...")
                else:
                    print(f"   ❌ Erro no download: {download_response.status_code}")
                    
            else:
                print(f"   ❌ Erro ao gerar URL de download: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Erro ao gerar URL de download: {e}")
    
    # 6. Listar arquivos
    print(f"\n6️⃣ Listando arquivos...")
    try:
        response = requests.get(
            f"{BASE_URL}/r2/files",
            params={"prefix": "test-uploads/"}
        )
        
        if response.status_code == 200:
            files_data = response.json()
            files = files_data.get('files', [])
            print(f"   ✅ Encontrados {len(files)} arquivo(s):")
            for file in files[:5]:  # Mostrar apenas os primeiros 5
                print(f"     📄 {file['key']} ({file['size']} bytes)")
        else:
            print(f"   ❌ Erro ao listar arquivos: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Erro ao listar arquivos: {e}")

    print(f"\n🎉 Teste concluído!")

if __name__ == "__main__":
    test_r2_functionality()