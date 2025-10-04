"""
Gerenciador de WebSockets para notificações em tempo real
"""
import json
import uuid
from typing import Dict, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Gerenciador de conexões WebSocket"""
    
    def __init__(self):
        # {analysis_id: {session_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # {session_id: analysis_id}
        self.session_to_analysis: Dict[str, str] = {}
        # Lock para operações concorrentes
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, analysis_id: str, client_id: str = None) -> str:
        """Conectar cliente WebSocket"""
        await websocket.accept()
        
        session_id = str(uuid.uuid4())
        
        async with self._lock:
            # Inicializar lista de conexões para análise se não existir
            if analysis_id not in self.active_connections:
                self.active_connections[analysis_id] = {}
            
            # Adicionar conexão
            self.active_connections[analysis_id][session_id] = websocket
            self.session_to_analysis[session_id] = analysis_id
        
        logger.info(f"WebSocket conectado: session_id={session_id}, analysis_id={analysis_id}, client_id={client_id}")
        
        # Enviar confirmação de conexão
        await self.send_personal_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return session_id
    
    async def disconnect(self, session_id: str):
        """Desconectar cliente WebSocket"""
        async with self._lock:
            if session_id in self.session_to_analysis:
                analysis_id = self.session_to_analysis[session_id]
                
                # Remover da lista de conexões
                if analysis_id in self.active_connections:
                    self.active_connections[analysis_id].pop(session_id, None)
                    
                    # Limpar análise se não há mais conexões
                    if not self.active_connections[analysis_id]:
                        del self.active_connections[analysis_id]
                
                # Remover mapeamento
                del self.session_to_analysis[session_id]
                
                logger.info(f"WebSocket desconectado: session_id={session_id}, analysis_id={analysis_id}")
    
    async def send_personal_message(self, session_id: str, message: dict):
        """Enviar mensagem para uma sessão específica"""
        if session_id not in self.session_to_analysis:
            return False
        
        analysis_id = self.session_to_analysis[session_id]
        
        if analysis_id in self.active_connections and session_id in self.active_connections[analysis_id]:
            websocket = self.active_connections[analysis_id][session_id]
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {session_id}: {e}")
                await self.disconnect(session_id)
                return False
        
        return False
    
    async def broadcast_to_analysis(self, analysis_id: str, message: dict, exclude_session: str = None):
        """Broadcast mensagem para todas as conexões de uma análise"""
        if analysis_id not in self.active_connections:
            return 0
        
        connections = self.active_connections[analysis_id].copy()
        sent_count = 0
        failed_sessions = []
        
        for session_id, websocket in connections.items():
            if exclude_session and session_id == exclude_session:
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
                sent_count += 1
            except Exception as e:
                logger.error(f"Erro ao enviar broadcast para {session_id}: {e}")
                failed_sessions.append(session_id)
        
        # Limpar conexões falhas
        for session_id in failed_sessions:
            await self.disconnect(session_id)
        
        return sent_count
    
    async def send_analysis_status(self, analysis_id: str, status: str, progress: float = None, message: str = None):
        """Enviar atualização de status da análise"""
        data = {
            "type": "analysis_status",
            "analysis_id": analysis_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            data["progress"] = progress
        if message:
            data["message"] = message
        
        await self.broadcast_to_analysis(analysis_id, data)
    
    async def send_analysis_completed(self, analysis_id: str, summary: dict = None):
        """Enviar notificação de análise completa"""
        data = {
            "type": "analysis_completed",
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if summary:
            data["summary"] = summary
        
        await self.broadcast_to_analysis(analysis_id, data)
    
    async def send_analysis_error(self, analysis_id: str, error_message: str):
        """Enviar notificação de erro na análise"""
        data = {
            "type": "analysis_error",
            "analysis_id": analysis_id,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_analysis(analysis_id, data)
    
    async def send_heartbeat(self, session_id: str):
        """Enviar heartbeat para manter conexão viva"""
        await self.send_personal_message(session_id, {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_active_sessions_for_analysis(self, analysis_id: str) -> List[str]:
        """Obter lista de sessões ativas para uma análise"""
        if analysis_id in self.active_connections:
            return list(self.active_connections[analysis_id].keys())
        return []
    
    def get_total_connections(self) -> int:
        """Obter número total de conexões ativas"""
        return len(self.session_to_analysis)
    
    def get_analyses_with_connections(self) -> List[str]:
        """Obter lista de análises com conexões ativas"""
        return list(self.active_connections.keys())
    
    async def cleanup_stale_connections(self):
        """Limpar conexões inativas (chamado periodicamente)"""
        stale_sessions = []
        
        for session_id, analysis_id in self.session_to_analysis.items():
            if analysis_id in self.active_connections and session_id in self.active_connections[analysis_id]:
                websocket = self.active_connections[analysis_id][session_id]
                try:
                    # Tentar enviar ping
                    await websocket.ping()
                except Exception:
                    stale_sessions.append(session_id)
            else:
                stale_sessions.append(session_id)
        
        # Remover conexões inativas
        for session_id in stale_sessions:
            await self.disconnect(session_id)
        
        return len(stale_sessions)


# Instância global do gerenciador de WebSocket
websocket_manager = WebSocketManager()


class WebSocketHandler:
    """Handler para endpoints WebSocket"""
    
    def __init__(self, manager: WebSocketManager):
        self.manager = manager
    
    async def handle_connection(self, websocket: WebSocket, analysis_id: str, client_id: str = None):
        """Gerenciar conexão WebSocket"""
        session_id = await self.manager.connect(websocket, analysis_id, client_id)
        
        try:
            while True:
                # Aguardar mensagens do cliente
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await self._handle_client_message(session_id, message)
                except json.JSONDecodeError:
                    await self.manager.send_personal_message(session_id, {
                        "type": "error",
                        "message": "Formato de mensagem inválido"
                    })
                
        except WebSocketDisconnect:
            logger.info(f"Cliente desconectado: {session_id}")
        except Exception as e:
            logger.error(f"Erro na conexão WebSocket {session_id}: {e}")
        finally:
            await self.manager.disconnect(session_id)
    
    async def _handle_client_message(self, session_id: str, message: dict):
        """Processar mensagem do cliente"""
        message_type = message.get("type")
        
        if message_type == "ping":
            await self.manager.send_heartbeat(session_id)
        
        elif message_type == "request_status":
            # Cliente solicita status atual da análise
            analysis_id = self.manager.session_to_analysis.get(session_id)
            if analysis_id:
                # Aqui você pode buscar o status atual do banco de dados
                # e enviar para o cliente
                await self.manager.send_personal_message(session_id, {
                    "type": "status_requested",
                    "analysis_id": analysis_id,
                    "message": "Status será enviado em breve"
                })
        
        else:
            await self.manager.send_personal_message(session_id, {
                "type": "error",
                "message": f"Tipo de mensagem não reconhecido: {message_type}"
            })


# Handler global para WebSocket
websocket_handler = WebSocketHandler(websocket_manager)