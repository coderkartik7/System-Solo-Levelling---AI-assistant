"""Main FastAPI application for AI Voice Conversational Agent."""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import json
import asyncio
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .models.schemas import (
    LLMResponse, ChatResponse, ChatHistoryResponse, SessionsResponse,
    ChatMessage, MessageRole, VoiceStyle, AudioFormat, HealthResponse,
    ServiceStatus
)
from .services.stt_service import stt_service, STTServiceError
from .services.tts_service import tts_service, TTSServiceError
from .services.llm_service import llm_service, LLMServiceError
from .utils.session_manager import session_manager

# Setup configuration
config.validate()
config.setup_directories()
config.setup_logging()

logger = logging.getLogger(__name__)

class ConnectionManager:
    """WebSocket connection manager for handling multiple clients."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_info: Dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept and store a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store client information
        self.client_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        logger.info(f"WebSocket client connected: {self.client_info[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            client_info = self.client_info.get(websocket, {})
            client_id = client_info.get("client_id", "unknown")
            
            self.active_connections.remove(websocket)
            if websocket in self.client_info:
                del self.client_info[websocket]
            
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(message)
            
            # Update message count
            if websocket in self.client_info:
                self.client_info[websocket]["message_count"] += 1
                
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, exclude_websocket: Optional[WebSocket] = None):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        
        for connection in self.active_connections:
            if connection != exclude_websocket:
                try:
                    await connection.send_text(message)
                    
                    # Update message count
                    if connection in self.client_info:
                        self.client_info[connection]["message_count"] += 1
                        
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected_clients.append(connection)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.disconnect(client)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.active_connections),
            "clients": [
                {
                    "client_id": info["client_id"],
                    "connected_at": info["connected_at"],
                    "message_count": info["message_count"]
                }
                for info in self.client_info.values()
            ]
        }

class AudioStreamManager:
    """Manages audio streaming sessions and file handling."""
    
    def __init__(self):
        self.active_streams: Dict[str, dict] = {}
        self.audio_files: Dict[str, Any] = {}
        
        # Create audio streams directory
        self.streams_dir = config.UPLOADS_DIR / "audio_streams"
        self.streams_dir.mkdir(exist_ok=True)
        
        logger.info("Audio Stream Manager initialized")
    
    def start_audio_stream(self, session_id: str, client_id: str = "unknown") -> str:
        """Start a new audio streaming session."""
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stream_id = f"stream_{timestamp}_{uuid.uuid4().hex[:8]}"
        filename = f"{stream_id}_{client_id}.webm"
        filepath = self.streams_dir / filename
        
        # Initialize stream info
        stream_info = {
            "stream_id": stream_id,
            "session_id": session_id,
            "client_id": client_id,
            "filepath": filepath,
            "filename": filename,
            "started_at": datetime.now().isoformat(),
            "chunks_received": 0,
            "total_bytes": 0,
            "is_active": True
        }
        
        # Open file for binary writing
        try:
            file_handle = open(filepath, "wb")
            self.audio_files[stream_id] = file_handle
            self.active_streams[stream_id] = stream_info
            
            logger.info(f"Started audio stream: {stream_id} -> {filepath}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise Exception(f"Could not start audio stream: {e}")
    
    def write_audio_chunk(self, stream_id: str, audio_data: bytes) -> bool:
        """Write audio chunk to the stream file."""
        
        if stream_id not in self.active_streams:
            logger.error(f"Stream {stream_id} not found")
            return False
        
        if stream_id not in self.audio_files:
            logger.error(f"File handle for stream {stream_id} not found")
            return False
        
        try:
            file_handle = self.audio_files[stream_id]
            file_handle.write(audio_data)
            file_handle.flush()  # Ensure data is written immediately
            
            # Update stream statistics
            stream_info = self.active_streams[stream_id]
            stream_info["chunks_received"] += 1
            stream_info["total_bytes"] += len(audio_data)
            
            logger.debug(f"Wrote {len(audio_data)} bytes to stream {stream_id} (chunk #{stream_info['chunks_received']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write audio chunk to stream {stream_id}: {e}")
            return False
    
    def stop_audio_stream(self, stream_id: str) -> dict:
        """Stop an audio streaming session."""
        
        if stream_id not in self.active_streams:
            return {"success": False, "error": "Stream not found"}
        
        try:
            # Close file handle
            if stream_id in self.audio_files:
                self.audio_files[stream_id].close()
                del self.audio_files[stream_id]
            
            # Update stream info
            stream_info = self.active_streams[stream_id]
            stream_info["is_active"] = False
            stream_info["ended_at"] = datetime.now().isoformat()
            
            # Get final statistics
            result = {
                "success": True,
                "stream_id": stream_id,
                "filepath": str(stream_info["filepath"]),
                "filename": stream_info["filename"],
                "chunks_received": stream_info["chunks_received"],
                "total_bytes": stream_info["total_bytes"],
                "duration": stream_info.get("ended_at", "N/A"),
                "file_exists": stream_info["filepath"].exists()
            }
            
            logger.info(f"Stopped audio stream {stream_id}: {result['chunks_received']} chunks, {result['total_bytes']} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Failed to stop audio stream {stream_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_stream_info(self, stream_id: str) -> Optional[dict]:
        """Get information about an active stream."""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id].copy()
        return None
    
    def cleanup_inactive_streams(self):
        """Clean up any streams that weren't properly closed."""
        cleanup_count = 0
        
        for stream_id in list(self.active_streams.keys()):
            stream_info = self.active_streams[stream_id]
            
            if stream_info["is_active"]:
                # Check if file handle still exists
                if stream_id in self.audio_files:
                    try:
                        # Try to close the file handle
                        self.audio_files[stream_id].close()
                        del self.audio_files[stream_id]
                        cleanup_count += 1
                        
                        # Mark as inactive
                        stream_info["is_active"] = False
                        stream_info["ended_at"] = datetime.now().isoformat()
                        
                    except Exception as e:
                        logger.error(f"Error cleaning up stream {stream_id}: {e}")
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} inactive audio streams")
    
    def get_all_streams(self) -> list:
        """Get information about all streams."""
        return list(self.active_streams.values())

# Initialize the connection manager (add this after your other global instances)
websocket_manager = ConnectionManager()
audio_stream_manager = AudioStreamManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code that runs on startup
    logger.info("🚀 Starting AI Voice Conversational Agent v2.0.0")
    logger.info(f"🌐 Server: http://{config.HOST}:{config.PORT}")
    logger.info(f"📖 API Docs: http://{config.HOST}:{config.PORT}/docs")

    logger.info(f"🔑 Services configured:")
    logger.info(f"   - STT (AssemblyAI): {'✓' if config.ASSEMBLY_API_KEY else '✗'}")
    logger.info(f"   - LLM (Gemini): {'✓' if config.GEMINI_API_KEY else '✗'}")
    logger.info(f"   - TTS (Murf): {'✓' if config.MURF_API_KEY else '✗'}")
    # The 'yield' pauses the function and lets the app run
    yield
    # Code that runs on shutdown
    logger.info("🛑 Shutting down AI Voice Conversational Agent")

# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="AI Voice Conversational Agent",
    description="An intelligent voice agent powered by Gemini, AssemblyAI, and Murf",
    version="2.0.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Serve the main application HTML page."""
    try:
        html_file = config.TEMPLATES_DIR / "index.html"
        with open(html_file, "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        logger.error(f"Template not found: {html_file}")
        raise HTTPException(status_code=404, detail="Application template not found")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    start_time = time.time()
    
    # Check all services
    services = []
    
    # STT Service
    stt_health = await stt_service.health_check()
    services.append(ServiceStatus(
        service="STT (AssemblyAI)",
        status=stt_health["status"],
        response_time=stt_health.get("response_time"),
        error=stt_health.get("error")
    ))
    
    # LLM Service
    llm_health = await llm_service.health_check()
    services.append(ServiceStatus(
        service="LLM (Gemini)",
        status=llm_health["status"],
        response_time=llm_health.get("response_time"),
        error=llm_health.get("error")
    ))
    
    # TTS Service
    tts_health = await tts_service.health_check()
    services.append(ServiceStatus(
        service="TTS (Murf)",
        status=tts_health["status"],
        response_time=tts_health.get("response_time"),
        error=tts_health.get("error")
    ))
    
    # Overall health
    all_healthy = all(service.status == "healthy" for service in services)
    uptime = time.time() - start_time
    
    return HealthResponse(
        success=all_healthy,
        services=services,
        uptime=uptime,
        message="All services healthy" if all_healthy else "Some services unhealthy"
    )

@app.post("/llm/query", response_model=LLMResponse)
async def llm_query(
    audio_file: UploadFile = File(...),
    model: str = Form(default=config.DEFAULT_MODEL)
):
    """
    Process audio through the complete LLM pipeline (STT -> LLM -> TTS).
    This is the original endpoint for backward compatibility.
    """
    start_time = time.time()
    
    if not audio_file:
        raise HTTPException(status_code=400, detail="Audio file is required")
    
    # Generate unique filename
    filename = f"llm_{int(time.time())}_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
    filepath = config.UPLOADS_DIR / filename
    
    try:
        # Save uploaded file
        logger.info(f"Processing LLM query: {audio_file.filename}")
        with open(filepath, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Step 1: Speech to Text
        transcription_result = await stt_service.transcribe_audio(filepath)
        logger.info(f"Transcribed: {transcription_result.text[:100]}...")
        
        # Step 2: Generate LLM response
        llm_result = await llm_service.generate_response(
            prompt=transcription_result.text,
            model=model
        )
        logger.info(f"LLM response: {llm_result.response[:100]}...")
        
        # Step 3: Text to Speech
        tts_result = await tts_service.synthesize_speech(
            text=llm_result.response
        )
        logger.info(f"Generated audio: {tts_result.audio_url}")
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            success=True,
            transcribed_text=transcription_result.text,
            llm_response=llm_result.response,
            audio_url=tts_result.audio_url,
            processing_time=processing_time,
            message="LLM pipeline completed successfully"
        )
        
    except (STTServiceError, LLMServiceError, TTSServiceError) as e:
        logger.error(f"Service error in LLM pipeline: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in LLM pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {e}")

@app.post("/agent/chat/{session_id}", response_model=ChatResponse)
async def chat_agent(
    session_id: str,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    model: str = Form(default=config.DEFAULT_MODEL),
    voice_id: str = Form(default=config.DEFAULT_VOICE_ID),
    voice_style: VoiceStyle = Form(default=VoiceStyle.CONVERSATIONAL),
    disable_thinking: bool = Form(default=False)
):
    """
    Chat with the AI agent using voice input and maintaining conversation history.
    """
    start_time = time.time()
    
    if not audio_file:
        raise HTTPException(status_code=400, detail="Audio file is required")
    
    # Generate unique filename
    filename = f"chat_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
    filepath = config.UPLOADS_DIR / filename
    
    try:
        # Save uploaded file
        logger.info(f"Processing chat for session: {session_id}")
        with open(filepath, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Ensure session exists
        session_manager.create_session(session_id)
        
        # Step 1: Speech to Text
        transcription_result = await stt_service.transcribe_audio(filepath)
        user_message = transcription_result.text
        
        # Add user message to session
        session_manager.add_message(
            session_id,
            ChatMessage(role=MessageRole.USER, content=user_message)
        )
        
        # Get conversation context
        chat_history = session_manager.get_recent_messages(session_id, count=config.MAX_CHAT_HISTORY)
        
        # Step 2: Generate LLM response
        llm_result = await llm_service.generate_response(
            prompt=user_message,
            chat_history=chat_history[:-1],  # Exclude the current user message
            model=model,
            disable_thinking=disable_thinking
        )
        
        # Add assistant response to session
        session_manager.add_message(
            session_id,
            ChatMessage(role=MessageRole.ASSISTANT, content=llm_result.response)
        )
        
        # Step 3: Text to Speech
        tts_result = await tts_service.synthesize_speech(
            text=llm_result.response,
            voice_id=voice_id,
            voice_style=voice_style
        )
        
        # Get updated chat history
        updated_history = session_manager.get_session_history(session_id)
        
        processing_time = time.time() - start_time
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, filepath)
        
        return ChatResponse(
            success=True,
            session_id=session_id,
            transcribed_text=user_message,
            llm_response=llm_result.response,
            audio_url=tts_result.audio_url,
            chat_history=updated_history,
            processing_time=processing_time,
            message="Chat completed successfully"
        )
        
    except (STTServiceError, LLMServiceError, TTSServiceError) as e:
        logger.error(f"Service error in chat agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/agent/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a specific session."""
    try:
        chat_history = session_manager.get_session_history(session_id)
        
        return ChatHistoryResponse(
            success=True,
            session_id=session_id,
            chat_history=chat_history,
            total_messages=len(chat_history),
            message="Chat history retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/agent/chat/{session_id}/history")
async def clear_chat_history(session_id: str):
    """Clear chat history for a specific session."""
    try:
        cleared = session_manager.clear_session(session_id)
        
        if cleared:
            return {"success": True, "message": "Chat history cleared successfully"}
        else:
            return {"success": True, "message": "Session not found (already cleared)"}
            
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/agent/sessions", response_model=SessionsResponse)
async def get_active_sessions():
    """Get all active chat sessions."""
    try:
        active_sessions = session_manager.get_active_sessions()
        
        return SessionsResponse(
            success=True,
            active_sessions=active_sessions,
            total_sessions=len(active_sessions),
            message="Active sessions retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving active sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/stats")
async def get_stats():
    """Get application statistics."""
    try:
        session_stats = session_manager.get_session_stats()
        
        return {
            "success": True,
            "stats": {
                **session_stats,
                "uptime": "N/A",  # Could implement uptime tracking
                "version": "2.0.0"
            },
            "message": "Statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Background task functions
def cleanup_file(filepath: Path):
    """Clean up temporary files."""
    try:
        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {e}")

# Legacy endpoint for backward compatibility
@app.get("/api/data")
async def get_data():
    """Legacy endpoint from Day 1 - for backward compatibility."""
    return {
        "message": "Hello from FastAPI backend!",
        "status": "success",
        "framework": "FastAPI",
        "version": "2.0.0"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Basic WebSocket endpoint for real-time communication.
    
    This endpoint accepts WebSocket connections and handles:
    - Echo messages back to the sender
    - Broadcast messages to all connected clients
    - Connection management
    """
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Try to parse as JSON for structured messages
                message_data = json.loads(data)
                
                # Handle different message types
                message_type = message_data.get("type", "echo")
                message_content = message_data.get("message", "")
                client_id = message_data.get("client_id", "anonymous")
                
                logger.info(f"Received WebSocket message: {message_type} from {client_id}")
                
                if message_type == "echo":
                    # Echo the message back to sender
                    response = {
                        "type": "echo_response",
                        "original_message": message_content,
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Echo: {message_content}"
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(response), websocket
                    )
                
                elif message_type == "broadcast":
                    # Broadcast message to all clients
                    broadcast_message = {
                        "type": "broadcast_message",
                        "from_client": client_id,
                        "message": message_content,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.broadcast(
                        json.dumps(broadcast_message), exclude_websocket=websocket
                    )
                    
                    # Send confirmation to sender
                    confirmation = {
                        "type": "broadcast_confirmation",
                        "message": f"Broadcasted to {len(websocket_manager.active_connections) - 1} clients",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(confirmation), websocket
                    )
                
                elif message_type == "ping":
                    # Respond to ping with pong
                    pong_response = {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                        "message": "pong"
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(pong_response), websocket
                    )
                
                elif message_type == "stats":
                    # Send connection statistics
                    stats = websocket_manager.get_connection_stats()
                    stats_response = {
                        "type": "stats_response",
                        "timestamp": datetime.now().isoformat(),
                        "stats": stats
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(stats_response), websocket
                    )
                
                else:
                    # Unknown message type - send error
                    error_response = {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.send_personal_message(
                        json.dumps(error_response), websocket
                    )
                    
            except json.JSONDecodeError:
                # Handle plain text messages
                logger.info(f"Received plain text WebSocket message: {data}")
                
                # Simple echo for plain text
                response_message = f"Echo: {data} (received at {datetime.now().isoformat()})"
                await websocket_manager.send_personal_message(response_message, websocket)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@app.websocket("/ws/audio/{session_id}")
async def audio_streaming_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming audio data from client to server.
    
    Protocol:
    1. Client connects with session_id
    2. Client sends 'start_stream' message to begin
    3. Client sends binary audio chunks
    4. Client sends 'stop_stream' message to end
    """
    client_id = f"audio_client_{session_id}"
    
    await websocket_manager.connect(websocket, client_id)
    
    current_stream_id = None
    
    # Send welcome message
    welcome_message = {
        "type": "audio_welcome",
        "session_id": session_id,
        "message": "Connected to audio streaming endpoint. Send 'start_stream' to begin.",
        "timestamp": datetime.now().isoformat()
    }
    await websocket_manager.send_personal_message(json.dumps(welcome_message), websocket)
    
    try:
        while True:
            # Receive message (could be text command or binary audio data)
            message = await websocket.receive()
            
            # Handle text messages (commands)
            if "text" in message:
                try:
                    command_data = json.loads(message["text"])
                    command_type = command_data.get("type")
                    
                    if command_type == "start_stream":
                        # Start new audio stream
                        try:
                            current_stream_id = audio_stream_manager.start_audio_stream(
                                session_id=session_id,
                                client_id=client_id
                            )
                            
                            response = {
                                "type": "stream_started",
                                "stream_id": current_stream_id,
                                "session_id": session_id,
                                "message": "Audio streaming started. Send binary audio data.",
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket_manager.send_personal_message(json.dumps(response), websocket)
                            
                        except Exception as e:
                            error_response = {
                                "type": "error",
                                "message": f"Failed to start audio stream: {str(e)}",
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket_manager.send_personal_message(json.dumps(error_response), websocket)
                    
                    elif command_type == "stop_stream":
                        # Stop current audio stream
                        if current_stream_id:
                            result = audio_stream_manager.stop_audio_stream(current_stream_id)
                            
                            response = {
                                "type": "stream_stopped",
                                "stream_id": current_stream_id,
                                "result": result,
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket_manager.send_personal_message(json.dumps(response), websocket)
                            
                            current_stream_id = None
                        else:
                            error_response = {
                                "type": "error",
                                "message": "No active stream to stop",
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket_manager.send_personal_message(json.dumps(error_response), websocket)
                    
                    elif command_type == "stream_info":
                        # Get current stream info
                        if current_stream_id:
                            stream_info = audio_stream_manager.get_stream_info(current_stream_id)
                            response = {
                                "type": "stream_info",
                                "stream_info": stream_info,
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            response = {
                                "type": "stream_info",
                                "message": "No active stream",
                                "timestamp": datetime.now().isoformat()
                            }
                        await websocket_manager.send_personal_message(json.dumps(response), websocket)
                    
                    else:
                        # Unknown command
                        error_response = {
                            "type": "error",
                            "message": f"Unknown command: {command_type}",
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket_manager.send_personal_message(json.dumps(error_response), websocket)
                        
                except json.JSONDecodeError:
                    # Handle plain text commands
                    command = message["text"].strip().lower()
                    
                    if command == "start":
                        current_stream_id = audio_stream_manager.start_audio_stream(
                            session_id=session_id,
                            client_id=client_id
                        )
                        response = f"Audio streaming started: {current_stream_id}"
                        await websocket_manager.send_personal_message(response, websocket)
                    
                    elif command == "stop":
                        if current_stream_id:
                            result = audio_stream_manager.stop_audio_stream(current_stream_id)
                            response = f"Audio streaming stopped: {result}"
                            current_stream_id = None
                        else:
                            response = "No active stream to stop"
                        await websocket_manager.send_personal_message(response, websocket)
                    
                    else:
                        response = f"Unknown command: {command}. Use 'start' or 'stop'"
                        await websocket_manager.send_personal_message(response, websocket)
            
            # Handle binary messages (audio data)
            elif "bytes" in message:
                if current_stream_id:
                    audio_data = message["bytes"]
                    success = audio_stream_manager.write_audio_chunk(current_stream_id, audio_data)
                    
                    if not success:
                        error_response = {
                            "type": "error",
                            "message": "Failed to write audio chunk",
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket_manager.send_personal_message(json.dumps(error_response), websocket)
                else:
                    # No active stream - send error
                    error_response = {
                        "type": "error",
                        "message": "No active stream. Send 'start_stream' first.",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.send_personal_message(json.dumps(error_response), websocket)
    
    except WebSocketDisconnect:
        # Clean up on disconnect
        if current_stream_id:
            result = audio_stream_manager.stop_audio_stream(current_stream_id)
            logger.info(f"Auto-stopped stream {current_stream_id} on client disconnect: {result}")
        
        websocket_manager.disconnect(websocket)
        logger.info(f"Audio streaming client {client_id} disconnected")
    
    except Exception as e:
        logger.error(f"Audio streaming error for {client_id}: {e}")
        
        # Clean up on error
        if current_stream_id:
            result = audio_stream_manager.stop_audio_stream(current_stream_id)
            logger.info(f"Auto-stopped stream {current_stream_id} on error: {result}")
        
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint_with_id(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint with client ID parameter.
    
    This allows clients to identify themselves with a custom ID.
    """
    await websocket_manager.connect(websocket, client_id)
    
    # Send welcome message
    welcome_message = {
        "type": "welcome",
        "client_id": client_id,
        "message": f"Welcome {client_id}! You are now connected to the WebSocket server.",
        "timestamp": datetime.now().isoformat(),
        "connected_clients": len(websocket_manager.active_connections)
    }
    await websocket_manager.send_personal_message(json.dumps(welcome_message), websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "echo")
                message_content = message_data.get("message", "")
                
                logger.info(f"Received message from {client_id}: {message_type}")
                
                # Add client_id to the message data
                message_data["from_client_id"] = client_id
                message_data["timestamp"] = datetime.now().isoformat()
                
                if message_type == "echo":
                    response = {
                        "type": "echo_response",
                        "to_client": client_id,
                        "original_message": message_content,
                        "message": f"Echo for {client_id}: {message_content}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.send_personal_message(json.dumps(response), websocket)
                
                elif message_type == "broadcast":
                    # Broadcast with client ID
                    broadcast_message = {
                        "type": "broadcast_message",
                        "from_client": client_id,
                        "message": message_content,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.broadcast(json.dumps(broadcast_message), exclude_websocket=websocket)
                    
                    # Confirmation
                    confirmation = {
                        "type": "broadcast_confirmation",
                        "client_id": client_id,
                        "message": f"Message broadcasted to {len(websocket_manager.active_connections) - 1} other clients",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.send_personal_message(json.dumps(confirmation), websocket)
                
                else:
                    # Echo back with client ID
                    response = {
                        "type": "response",
                        "client_id": client_id,
                        "original_type": message_type,
                        "message": f"Received {message_type} from {client_id}: {message_content}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket_manager.send_personal_message(json.dumps(response), websocket)
                    
            except json.JSONDecodeError:
                # Handle plain text
                response_message = f"[{client_id}] Echo: {data} (at {datetime.now().isoformat()})"
                await websocket_manager.send_personal_message(response_message, websocket)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(websocket)

# Add REST endpoint to get WebSocket statistics
@app.get("/api/websocket/stats")
async def get_websocket_stats():
    """Get current WebSocket connection statistics."""
    stats = websocket_manager.get_connection_stats()
    return {
        "success": True,
        "websocket_stats": stats,
        "message": "WebSocket statistics retrieved successfully"
    }

# Add REST endpoint to broadcast message to all WebSocket clients
@app.post("/api/websocket/broadcast")
async def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients via REST API."""
    if not message.get("message"):
        raise HTTPException(status_code=400, detail="Message content is required")
    
    broadcast_data = {
        "type": "server_broadcast",
        "message": message["message"],
        "timestamp": datetime.now().isoformat(),
        "source": "REST API"
    }
    
    await websocket_manager.broadcast(json.dumps(broadcast_data))
    
    return {
        "success": True,
        "message": f"Message broadcasted to {len(websocket_manager.active_connections)} clients",
        "clients_reached": len(websocket_manager.active_connections)
    }


@app.get("/api/audio/streams")
async def get_audio_streams():
    """Get information about all audio streams."""
    streams = audio_stream_manager.get_all_streams()
    
    return {
        "success": True,
        "total_streams": len(streams),
        "streams": streams,
        "message": "Audio streams retrieved successfully"
    }

@app.get("/api/audio/streams/{stream_id}")
async def get_audio_stream_info(stream_id: str):
    """Get information about a specific audio stream."""
    stream_info = audio_stream_manager.get_stream_info(stream_id)
    
    if stream_info:
        return {
            "success": True,
            "stream_info": stream_info,
            "message": "Stream info retrieved successfully"
        }
    else:
        raise HTTPException(status_code=404, detail="Stream not found")

@app.post("/api/audio/cleanup")
async def cleanup_audio_streams():
    """Clean up inactive audio streams."""
    try:
        audio_stream_manager.cleanup_inactive_streams()
        
        return {
            "success": True,
            "message": "Audio streams cleaned up successfully"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup audio streams: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup streams")

if __name__ == "__main__":
    print("🚀 Starting Enhanced AI Voice Conversational Agent v2.0.0...")
    print(f"🌐 Visit: http://{config.HOST}:{config.PORT}")
    print(f"📖 API Docs: http://{config.HOST}:{config.PORT}/docs")
    print(f"🔧 Debug Mode: {config.DEBUG}")
    
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )