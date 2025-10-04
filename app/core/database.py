"""
Configuração e factory do banco de dados
"""
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.models.database import Base


class DatabaseConfig:
    """Configuração do banco de dados"""
    
    def __init__(self):
        # Garantir que o diretório data existe
        data_dir = os.path.join(os.getcwd(), "app", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        self.database_url = os.getenv(
            "DATABASE_URL", 
            f"sqlite+aiosqlite:///{data_dir}/eda_backend.db"
        )
        self.echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
        self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
    
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")
    
    @property
    def is_postgresql(self) -> bool:
        return self.database_url.startswith("postgresql")


class DatabaseManager:
    """Gerenciador do banco de dados"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        
        # Configurações específicas para SQLite vs PostgreSQL
        engine_kwargs = {"echo": self.config.echo}
        
        if not self.config.is_sqlite:
            engine_kwargs.update({
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_pre_ping": True,
                "pool_recycle": 3600
            })
        
        self.engine = create_async_engine(
            self.config.database_url,
            **engine_kwargs
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self):
        """Criar todas as tabelas"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Remover todas as tabelas (apenas para desenvolvimento)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Obter sessão do banco de dados"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Fechar conexões do banco de dados"""
        await self.engine.dispose()


# Instância global do gerenciador de banco
db_manager = DatabaseManager()


# Dependency para FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency para obter sessão do banco de dados"""
    async for session in db_manager.get_session():
        yield session