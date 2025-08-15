"""Pydantic models for request/response validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    """Chat message model."""
    role: MessageRole
    content: str = Field(..., min_length=1)
    timestamp: Optional[str] = None

class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "MP3"
    WAV = "WAV" 
    WEBM = "WEBM"

class VoiceStyle(str, Enum):
    """Available voice styles."""
    CONVERSATIONAL = "Conversational"
    PROFESSIONAL = "Professional"
    CASUAL = "Casual"
    ENERGETIC = "Energetic"

class BaseResponse(BaseModel):
    """Base response model."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class LLMRequest(BaseModel):
    """Request model for LLM endpoint."""
    model: str = Field(default="gemini-2.0-flash-exp")
    disable_thinking: bool = Field(default=False)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    
    @validator('model')
    def validate_model(cls, v):
        allowed_models = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {allowed_models}")
        return v

class LLMResponse(BaseResponse):
    """Response model for LLM endpoint."""
    transcribed_text: Optional[str] = None
    llm_response: Optional[str] = None
    audio_url: Optional[str] = None
    processing_time: Optional[float] = None

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    model: str = Field(default="gemini-2.0-flash-exp")
    disable_thinking: bool = Field(default=False)
    voice_id: str = Field(default="en-US-daniel")
    voice_style: VoiceStyle = Field(default=VoiceStyle.CONVERSATIONAL)
    audio_format: AudioFormat = Field(default=AudioFormat.MP3)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

class ChatResponse(BaseResponse):
    """Response model for chat endpoint."""
    session_id: str
    transcribed_text: Optional[str] = None
    llm_response: Optional[str] = None
    audio_url: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None
    processing_time: Optional[float] = None

class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str
    message_count: int
    created_at: Optional[str] = None
    last_updated: Optional[str] = None

class SessionsResponse(BaseResponse):
    """Response model for sessions endpoint."""
    active_sessions: List[SessionInfo]
    total_sessions: int

class ChatHistoryResponse(BaseResponse):
    """Response model for chat history endpoint."""
    session_id: str
    chat_history: List[ChatMessage]
    total_messages: int

class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: str
    error_details: Optional[ErrorDetail] = None

class AudioUpload(BaseModel):
    """Audio upload validation."""
    filename: str
    size: int
    content_type: str
    
    @validator('size')
    def validate_size(cls, v):
        max_size = 50 * 1024 * 1024  # 50MB
        if v > max_size:
            raise ValueError(f"File size exceeds maximum allowed size of {max_size} bytes")
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = [
            "audio/webm",
            "audio/mp3", 
            "audio/mpeg",
            "audio/wav",
            "audio/mp4",
            "audio/ogg"
        ]
        if v not in allowed_types:
            raise ValueError(f"Content type must be one of: {allowed_types}")
        return v

class ServiceStatus(BaseModel):
    """Service status model."""
    service: str
    status: str
    response_time: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseResponse):
    """Health check response."""
    services: List[ServiceStatus]
    uptime: float
    version: str = "2.0.0"

# Request/Response models for specific endpoints
class TranscriptionResult(BaseModel):
    """Transcription service result."""
    text: str
    confidence: Optional[float] = None
    processing_time: float

class TTSResult(BaseModel):
    """TTS service result."""
    audio_url: str
    duration: Optional[float] = None
    format: str
    processing_time: float

class LLMResult(BaseModel):
    """LLM service result."""
    response: str
    model_used: str
    tokens_used: Optional[int] = None
    processing_time: float