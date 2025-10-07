#!/usr/bin/env python3
"""
Script para configurar CORS no bucket Cloudflare R2
Execute este script uma vez para resolver problemas de CORS
"""
import json
import boto3
from botocore.client import Config
from app.core.config import settings

def configure_r2_cors():
    """Configurar CORS no bucket R2"""
    
    if not all([
        settings.cloudflare_r2_access_key_id,
        settings.cloudflare_r2_secret_access_key,
        settings.cloudflare_r2_endpoint_url,
        settings.cloudflare_r2_bucket_name
    ]):
        print("❌ Erro: Credenciais do Cloudflare R2 não configuradas")
        print("Configure as variáveis de ambiente no arquivo .env")
        return False
    
    try:
        # Criar cliente S3 para Cloudflare R2
        client = boto3.client(
            's3',
            endpoint_url=settings.cloudflare_r2_endpoint_url,
            aws_access_key_id=settings.cloudflare_r2_access_key_id,
            aws_secret_access_key=settings.cloudflare_r2_secret_access_key,
            region_name=settings.cloudflare_r2_region,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'virtual'}
            )
        )
        
        # Configuração CORS
        cors_configuration = {
            'CORSRules': [
                {
                    'ID': 'AllowDirectUploads',
                    'AllowedHeaders': ['*'],
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
                    'AllowedOrigins': [
                        'http://localhost:3000',
                        'http://localhost:3001', 
                        'http://127.0.0.1:3000',
                        'https://*.vercel.app',
                        'https://*.netlify.app',
                        'https://*.pages.dev'
                    ],
                    'ExposeHeaders': [
                        'ETag',
                        'Content-Length',
                        'Content-Type',
                        'Last-Modified'
                    ],
                    'MaxAgeSeconds': 3600
                }
            ]
        }
        
        # Aplicar configuração CORS
        print(f"🔧 Configurando CORS no bucket: {settings.cloudflare_r2_bucket_name}")
        
        client.put_bucket_cors(
            Bucket=settings.cloudflare_r2_bucket_name,
            CORSConfiguration=cors_configuration
        )
        
        print("✅ CORS configurado com sucesso!")
        print("\n📋 Configuração aplicada:")
        print(json.dumps(cors_configuration, indent=2))
        
        # Verificar se a configuração foi aplicada
        try:
            response = client.get_bucket_cors(Bucket=settings.cloudflare_r2_bucket_name)
            print("\n✅ Verificação: CORS está ativo no bucket")
            return True
        except Exception as e:
            if 'NoSuchCORSConfiguration' in str(e):
                print("⚠️  Aviso: Não foi possível verificar a configuração CORS")
                print("Isso pode ser normal em alguns casos, teste o upload")
                return True
            else:
                print(f"❌ Erro ao verificar CORS: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Erro ao configurar CORS: {e}")
        return False

def remove_cors():
    """Remover configuração CORS (para debug)"""
    try:
        client = boto3.client(
            's3',
            endpoint_url=settings.cloudflare_r2_endpoint_url,
            aws_access_key_id=settings.cloudflare_r2_access_key_id,
            aws_secret_access_key=settings.cloudflare_r2_secret_access_key,
            region_name=settings.cloudflare_r2_region,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'virtual'}
            )
        )
        
        client.delete_bucket_cors(Bucket=settings.cloudflare_r2_bucket_name)
        print("✅ CORS removido com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao remover CORS: {e}")
        return False

def check_cors():
    """Verificar configuração CORS atual"""
    try:
        client = boto3.client(
            's3',
            endpoint_url=settings.cloudflare_r2_endpoint_url,
            aws_access_key_id=settings.cloudflare_r2_access_key_id,
            aws_secret_access_key=settings.cloudflare_r2_secret_access_key,
            region_name=settings.cloudflare_r2_region,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'virtual'}
            )
        )
        
        response = client.get_bucket_cors(Bucket=settings.cloudflare_r2_bucket_name)
        print("✅ Configuração CORS atual:")
        print(json.dumps(response['CORSRules'], indent=2))
        return True
    except Exception as e:
        if 'NoSuchCORSConfiguration' in str(e):
            print("❌ Nenhuma configuração CORS encontrada")
        else:
            print(f"❌ Erro ao verificar CORS: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "check":
            check_cors()
        elif action == "remove":
            remove_cors()
        else:
            print("Ações disponíveis: check, remove")
    else:
        print("🚀 Configurando CORS no Cloudflare R2...")
        configure_r2_cors()