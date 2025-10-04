"""
Serviço para upload de arquivos no Cloudflare R2 usando URLs pré-assinadas
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.client import Config

from .config import settings

logger = logging.getLogger(__name__)


class CloudflareR2Service:
    """Serviço para gerenciar uploads no Cloudflare R2"""
    
    def __init__(self):
        """Inicializar o cliente S3 para Cloudflare R2"""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializar cliente boto3 para Cloudflare R2"""
        try:
            if not all([
                settings.cloudflare_r2_access_key_id,
                settings.cloudflare_r2_secret_access_key,
                settings.cloudflare_r2_endpoint_url,
                settings.cloudflare_r2_bucket_name
            ]):
                logger.warning("Credenciais do Cloudflare R2 não configuradas")
                return
            
            self.client = boto3.client(
                's3',
                endpoint_url=settings.cloudflare_r2_endpoint_url,
                aws_access_key_id=settings.cloudflare_r2_access_key_id,
                aws_secret_access_key=settings.cloudflare_r2_secret_access_key,
                region_name=settings.cloudflare_r2_region,
                config=Config(
                    signature_version='s3v4',
                    s3={
                        'addressing_style': 'virtual'
                    },
                    retries={
                        'max_attempts': 3,
                        'mode': 'adaptive'
                    }
                )
            )
            
            logger.info("Cliente Cloudflare R2 inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente R2: {e}")
            self.client = None
    
    def is_configured(self) -> bool:
        """Verificar se o serviço está configurado corretamente"""
        return self.client is not None
    
    def generate_file_key(self, filename: str, folder: str = "uploads") -> str:
        """
        Gerar chave única para o arquivo
        
        Args:
            filename: Nome original do arquivo
            folder: Pasta onde salvar o arquivo
            
        Returns:
            Chave única para o arquivo
        """
        # Gerar ID único
        file_id = str(uuid.uuid4())
        
        # Extrair extensão do arquivo
        file_extension = Path(filename).suffix
        
        # Criar timestamp
        timestamp = datetime.now().strftime("%Y/%m/%d")
        
        # Criar chave única
        key = f"{folder}/{timestamp}/{file_id}{file_extension}"
        
        return key
    
    def generate_presigned_upload_url(
        self, 
        filename: str, 
        content_type: str = "application/octet-stream",
        folder: str = "uploads"
    ) -> Dict[str, Any]:
        """
        Gerar URL pré-assinada para upload usando PUT
        
        R2 não suporta presigned POST, então usamos PUT
        Simplificado para evitar problemas de assinatura com metadados
        
        Args:
            filename: Nome do arquivo
            content_type: Tipo de conteúdo do arquivo
            folder: Pasta de destino
            
        Returns:
            Dicionário com URL pré-assinada e metadados
        """
        if not self.is_configured():
            raise ValueError("Cloudflare R2 não está configurado")
        
        try:
            # Gerar chave única para o arquivo
            file_key = self.generate_file_key(filename, folder)
            
            # Gerar URL pré-assinada para PUT (sem metadados customizados)
            upload_url = self.client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': settings.cloudflare_r2_bucket_name,
                    'Key': file_key,
                    'ContentType': content_type
                },
                ExpiresIn=settings.presigned_url_expiration_seconds
            )
            
            return {
                "success": True,
                "upload_url": upload_url,
                "method": "PUT",
                "file_key": file_key,
                "content_type": content_type,
                "expires_in": settings.presigned_url_expiration_seconds,
                "max_file_size_mb": settings.max_file_size_mb,
                "expires_at": (
                    datetime.now() + timedelta(seconds=settings.presigned_url_expiration_seconds)
                ).isoformat(),
                "headers": {
                    "Content-Type": content_type
                },
                "original_filename": filename,
                "upload_timestamp": datetime.now().isoformat()
            }
            
        except ClientError as e:
            logger.error(f"Erro do cliente AWS: {e}")
            raise ValueError(f"Erro ao gerar URL pré-assinada: {e}")
        
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise ValueError(f"Erro interno: {e}")
    
    def generate_presigned_download_url(self, file_key: str) -> Dict[str, Any]:
        """
        Gerar URL pré-assinada para download
        
        Args:
            file_key: Chave do arquivo no R2
            
        Returns:
            Dicionário com URL pré-assinada para download
        """
        if not self.is_configured():
            raise ValueError("Cloudflare R2 não está configurado")
        
        try:
            download_url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.cloudflare_r2_bucket_name,
                    'Key': file_key
                },
                ExpiresIn=settings.presigned_url_expiration_seconds
            )
            
            return {
                "success": True,
                "download_url": download_url,
                "file_key": file_key,
                "expires_in": settings.presigned_url_expiration_seconds,
                "expires_at": (
                    datetime.now() + timedelta(seconds=settings.presigned_url_expiration_seconds)
                ).isoformat()
            }
            
        except ClientError as e:
            logger.error(f"Erro do cliente AWS: {e}")
            raise ValueError(f"Erro ao gerar URL de download: {e}")
        
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise ValueError(f"Erro interno: {e}")
    
    def get_file_info(self, file_key: str) -> Dict[str, Any]:
        """
        Obter informações de um arquivo no R2
        
        Args:
            file_key: Chave do arquivo
            
        Returns:
            Informações do arquivo
        """
        if not self.is_configured():
            raise ValueError("Cloudflare R2 não está configurado")
        
        try:
            response = self.client.head_object(
                Bucket=settings.cloudflare_r2_bucket_name,
                Key=file_key
            )
            
            return {
                "success": True,
                "file_key": file_key,
                "size": response.get('ContentLength', 0),
                "content_type": response.get('ContentType', ''),
                "last_modified": response.get('LastModified', '').isoformat() if response.get('LastModified') else '',
                "etag": response.get('ETag', '').strip('"'),
                "metadata": response.get('Metadata', {}),
                "original_filename": response.get('Metadata', {}).get('original-filename', '')
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise ValueError(f"Arquivo não encontrado: {file_key}")
            else:
                logger.error(f"Erro do cliente AWS: {e}")
                raise ValueError(f"Erro ao obter informações do arquivo: {e}")
        
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise ValueError(f"Erro interno: {e}")
    
    def download_file(self, file_key: str) -> bytes:
        """
        Baixar arquivo do R2
        
        Args:
            file_key: Chave do arquivo
            
        Returns:
            Conteúdo do arquivo em bytes
        """
        if not self.is_configured():
            raise ValueError("Cloudflare R2 não está configurado")
        
        try:
            logger.info(f"Baixando arquivo: {file_key}")
            
            response = self.client.get_object(
                Bucket=settings.cloudflare_r2_bucket_name,
                Key=file_key
            )
            
            # Ler conteúdo do arquivo
            file_content = response['Body'].read()
            
            logger.info(f"Arquivo baixado com sucesso: {file_key} ({len(file_content)} bytes)")
            return file_content
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise ValueError(f"Arquivo não encontrado: {file_key}")
            else:
                logger.error(f"Erro do cliente AWS: {e}")
                raise ValueError(f"Erro ao baixar arquivo: {e}")
        
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise ValueError(f"Erro interno: {e}")
    
    def delete_file(self, file_key: str) -> Dict[str, Any]:
        """
        Deletar arquivo do R2
        
        Args:
            file_key: Chave do arquivo
            
        Returns:
            Confirmação da deleção
        """
        if not self.is_configured():
            raise ValueError("Cloudflare R2 não está configurado")
        
        try:
            self.client.delete_object(
                Bucket=settings.cloudflare_r2_bucket_name,
                Key=file_key
            )
            
            return {
                "success": True,
                "message": f"Arquivo {file_key} deletado com sucesso",
                "file_key": file_key
            }
            
        except ClientError as e:
            logger.error(f"Erro do cliente AWS: {e}")
            raise ValueError(f"Erro ao deletar arquivo: {e}")
        
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise ValueError(f"Erro interno: {e}")
    
    def list_files(self, folder: str = "uploads", limit: int = 100) -> Dict[str, Any]:
        """
        Listar arquivos em uma pasta
        
        Args:
            folder: Pasta a ser listada
            limit: Limite de arquivos retornados
            
        Returns:
            Lista de arquivos
        """
        if not self.is_configured():
            raise ValueError("Cloudflare R2 não está configurado")
        
        try:
            response = self.client.list_objects_v2(
                Bucket=settings.cloudflare_r2_bucket_name,
                Prefix=f"{folder}/",
                MaxKeys=limit
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "etag": obj['ETag'].strip('"')
                })
            
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "folder": folder,
                "truncated": response.get('IsTruncated', False)
            }
            
        except ClientError as e:
            logger.error(f"Erro do cliente AWS: {e}")
            raise ValueError(f"Erro ao listar arquivos: {e}")
        
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise ValueError(f"Erro interno: {e}")


# Instância global do serviço
r2_service = CloudflareR2Service()