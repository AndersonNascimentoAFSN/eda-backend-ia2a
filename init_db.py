"""
Script de inicialização do banco de dados
"""
import asyncio
import os
from app.core.database import db_manager


async def init_database():
    """Inicializar banco de dados"""
    print("🔧 Inicializando banco de dados...")
    
    try:
        # Criar diretório de dados se não existir
        data_dir = "./app/data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"📁 Diretório {data_dir} criado")
        
        # Criar tabelas
        await db_manager.create_tables()
        print("✅ Tabelas criadas com sucesso")
        
        # Teste de conectividade
        async for session in db_manager.get_session():
            print("✅ Conexão com banco de dados testada")
            break
        
        print("🎯 Banco de dados inicializado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao inicializar banco de dados: {e}")
        raise


async def reset_database():
    """Resetar banco de dados (apenas para desenvolvimento)"""
    print("⚠️  RESETANDO banco de dados...")
    
    try:
        await db_manager.drop_tables()
        print("🗑️  Tabelas removidas")
        
        await db_manager.create_tables()
        print("✅ Tabelas recriadas")
        
        print("🎯 Banco de dados resetado com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao resetar banco de dados: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        asyncio.run(reset_database())
    else:
        asyncio.run(init_database())