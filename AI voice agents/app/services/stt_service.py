"""Speech-to-Text service using AssemblyAI."""

import logging
import time
from pathlib import Path
from typing import Optional
import assemblyai as aai

from ..config import config
from ..models.schemas import TranscriptionResult

logger = logging.getLogger(__name__)

class STTServiceError(Exception):
    """Custom exception for STT service errors."""
    pass

class STTService:
    """Speech-to-Text service wrapper."""
    
    def __init__(self):
        """Initialize STT service."""
        if not config.ASSEMBLY_API_KEY:
            raise STTServiceError("AssemblyAI API key not configured")
        
        aai.settings.api_key = config.ASSEMBLY_API_KEY
        self.transcriber = aai.Transcriber()
        logger.info("STT Service initialized with AssemblyAI")
    
    async def transcribe_audio(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult with transcribed text and metadata
            
        Raises:
            STTServiceError: If transcription fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting transcription for: {audio_path.name}")
            
            # Validate file exists
            if not audio_path.exists():
                raise STTServiceError(f"Audio file not found: {audio_path}")
            
            # Validate file size
            file_size = audio_path.stat().st_size
            if file_size == 0:
                raise STTServiceError("Audio file is empty")
            
            if file_size > config.MAX_FILE_SIZE:
                raise STTServiceError(f"Audio file too large: {file_size} bytes")
            
            logger.debug(f"Transcribing file: {audio_path} ({file_size} bytes)")
            
            # Perform transcription
            transcript = self.transcriber.transcribe(str(audio_path))
            
            # Check for transcription errors
            if transcript.status == aai.TranscriptStatus.error:
                error_msg = f"Transcription failed: {transcript.error}"
                logger.error(error_msg)
                raise STTServiceError(error_msg)
            
            # Validate transcription result
            transcribed_text = transcript.text or ""
            
            if not transcribed_text or transcribed_text.strip() == "":
                raise STTServiceError("No speech detected in audio")
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Transcription successful: {len(transcribed_text)} characters "
                f"in {processing_time:.2f}s"
            )
            
            return TranscriptionResult(
                text=transcribed_text.strip(),
                confidence=transcript.confidence if hasattr(transcript, 'confidence') else None,
                processing_time=processing_time
            )
            
        except STTServiceError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"STT service error: {str(e)}"
            logger.error(f"{error_msg} (after {processing_time:.2f}s)")
            raise STTServiceError(error_msg)
    
    def is_available(self) -> bool:
        """
        Check if STT service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            # Simple availability check
            return bool(config.ASSEMBLY_API_KEY and self.transcriber)
        except Exception as e:
            logger.error(f"STT availability check failed: {e}")
            return False
    
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported audio formats.
        
        Returns:
            List of supported file extensions
        """
        return [".mp3", ".wav", ".webm", ".m4a", ".ogg", ".flac", ".mp4"]
    
    async def health_check(self) -> dict:
        """
        Perform health check on STT service.
        
        Returns:
            Dictionary with health status
        """
        try:
            start_time = time.time()
            
            # Check if service is configured
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "error": "Service not properly configured",
                    "response_time": time.time() - start_time
                }
            
            # Additional health checks could go here
            # For now, just check configuration
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "supported_formats": self.get_supported_formats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "response_time": time.time() - start_time
            }

# Global STT service instance
stt_service = STTService()