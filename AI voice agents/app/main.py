"""Main FastAPI application for AI Voice Conversational Agent."""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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