"""
Aplicação principal FastAPI para EDA Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router
from app.api.enhanced_endpoints import router as enhanced_router
from app.api.cors_proxy import cors_router
from app.core.database import db_manager

# Criar aplicação FastAPI
app = FastAPI(
    title="EDA Backend API",
    description="API para análise exploratória de dados (EDA) de arquivos CSV usando ydata-profiling",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rotas da API
app.include_router(api_router, prefix="/api/v1", tags=["EDA"])
app.include_router(enhanced_router, tags=["Enhanced EDA"])
app.include_router(cors_router, tags=["CORS Proxy"])

# Endpoint raiz
@app.get("/")
async def root():
    """Endpoint raiz da API"""
    return {
        "message": "EDA Backend API",
        "version": "0.1.0",
        "description": "API para análise exploratória de dados de arquivos CSV",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "upload_csv": "/api/v1/upload-csv",
            "csv_info": "/api/v1/csv-info",
            "supported_formats": "/api/v1/supported-formats",
            "health": "/api/v1/health",
            "cors_proxy": "/cors-proxy/test-cors"
        }
    }


# Eventos de startup e shutdown
@app.on_event("startup")
async def startup_event():
    """Inicializar banco de dados e outros recursos"""
    try:
        await db_manager.create_tables()
        print("✅ Banco de dados inicializado")
    except Exception as e:
        print(f"⚠️ Erro ao inicializar banco de dados: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpar recursos"""
    try:
        await db_manager.close()
        print("✅ Conexões do banco de dados fechadas")
    except Exception as e:
        print(f"⚠️ Erro ao fechar banco de dados: {e}")


# Handler global de exceções
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": "Erro na requisição",
            "error": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Erro interno do servidor",
            "error": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)