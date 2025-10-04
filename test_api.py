#!/usr/bin/env python3
"""
Script de teste para a API EDA Backend
"""
import requests
import json
import time
import subprocess
import signal
import os

def test_api():
    """Testa os endpoints da API"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testando API EDA Backend...")
    
    # Testar endpoint raiz
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ GET / - Status: {response.status_code}")
        print(f"   Resposta: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Erro no endpoint raiz: {e}")
        return False
    
    # Testar health check
    try:
        response = requests.get(f"{base_url}/api/v1/health")
        print(f"✅ GET /api/v1/health - Status: {response.status_code}")
        print(f"   Resposta: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Erro no health check: {e}")
    
    # Testar formatos suportados
    try:
        response = requests.get(f"{base_url}/api/v1/supported-formats")
        print(f"✅ GET /api/v1/supported-formats - Status: {response.status_code}")
        print(f"   Resposta: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Erro nos formatos suportados: {e}")
    
    # Testar upload de CSV
    try:
        with open("example_data.csv", "rb") as f:
            files = {"file": ("example_data.csv", f, "text/csv")}
            response = requests.post(f"{base_url}/api/v1/upload-csv", files=files)
        
        print(f"✅ POST /api/v1/upload-csv - Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Sucesso: {result['success']}")
            print(f"   Arquivo: {result['filename']}")
            print(f"   Linhas processadas: {result['eda_data']['rows']}")
            print(f"   Colunas processadas: {result['eda_data']['columns']}")
            print(f"   Colunas: {result['eda_data']['column_names']}")
        else:
            print(f"   Erro: {response.text}")
    except Exception as e:
        print(f"❌ Erro no upload de CSV: {e}")
    
    # Testar informações básicas do CSV
    try:
        with open("example_data.csv", "rb") as f:
            files = {"file": ("example_data.csv", f, "text/csv")}
            response = requests.post(f"{base_url}/api/v1/csv-info", files=files)
        
        print(f"✅ POST /api/v1/csv-info - Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Resposta: {json.dumps(result, indent=2)}")
        else:
            print(f"   Erro: {response.text}")
    except Exception as e:
        print(f"❌ Erro nas informações do CSV: {e}")
    
    print("\n🎉 Testes concluídos!")
    return True

if __name__ == "__main__":
    test_api()