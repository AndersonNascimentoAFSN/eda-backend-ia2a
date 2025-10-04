"""
Configurações da aplicação usando Pydantic Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configurações da aplicação"""
    
    # Cloudflare R2 Configuration
    cloudflare_r2_access_key_id: str = ""
    cloudflare_r2_secret_access_key: str = ""
    cloudflare_r2_endpoint_url: str = ""
    cloudflare_r2_bucket_name: str = ""
    cloudflare_r2_region: str = "auto"
    
    # Upload Settings
    max_file_size_mb: int = 100
    presigned_url_expiration_seconds: int = 3600
    
    # API Settings
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instância global das configurações
settings = Settings()