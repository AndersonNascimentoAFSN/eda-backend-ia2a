"""
Endpoint proxy para contornar problemas de CORS temporariamente
Use apenas durante desenvolvimento
"""
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Router específico para proxy CORS
cors_router = APIRouter(prefix="/cors-proxy", tags=["CORS Proxy"])

@cors_router.options("/{full_path:path}")
async def cors_preflight(full_path: str):
    """Handle CORS preflight requests"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

@cors_router.put("/r2-upload")
async def proxy_r2_upload(request: Request):
    """
    Proxy para upload direto ao R2 com headers CORS
    URL: /cors-proxy/r2-upload?target_url=<R2_PRESIGNED_URL>
    """
    try:
        # Obter URL de destino dos query parameters
        target_url = request.query_params.get("target_url")
        if not target_url:
            raise HTTPException(status_code=400, detail="target_url é obrigatório")
        
        # Obter headers da requisição original (exceto host)
        headers = dict(request.headers)
        headers.pop("host", None)
        
        # Obter corpo da requisição
        body = await request.body()
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Fazer upload para R2
            response = await client.put(
                target_url,
                content=body,
                headers=headers
            )
            
            # Retornar resposta com headers CORS
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": "*"
                }
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Timeout no upload")
    except httpx.RequestError as e:
        logger.error(f"Erro no proxy upload: {e}")
        raise HTTPException(status_code=502, detail=f"Erro no proxy: {str(e)}")
    except Exception as e:
        logger.error(f"Erro inesperado no proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@cors_router.get("/test-cors")
async def test_cors():
    """Testar se CORS está funcionando"""
    return {
        "message": "CORS está funcionando!",
        "timestamp": "2024-01-01T00:00:00Z",
        "headers_sent": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*"
        }
    }