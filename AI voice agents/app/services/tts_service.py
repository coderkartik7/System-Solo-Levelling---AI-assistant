"""Text-to-Speech service using Murf AI."""

import logging
import time
from typing import Optional
import requests
from typing import Optional
from ..config import config
from ..models.schemas import TTSResult, VoiceStyle, AudioFormat

logger = logging.getLogger(__name__)

class TTSServiceError(Exception):
    """Custom exception for TTS service errors."""
    pass

class TTSService:
    """Text-to-Speech service wrapper."""
    
    def __init__(self):
        """Initialize TTS service."""
        if not config.MURF_API_KEY:
            raise TTSServiceError("Murf API key not configured")
        
        self.api_key = config.MURF_API_KEY
        self.api_url = config.MURF_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "api-key": self.api_key,
            "Content-Type": "application/json"
        })
        
        logger.info("TTS Service initialized with Murf AI")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_style: VoiceStyle = VoiceStyle.CONVERSATIONAL,
        audio_format: AudioFormat = AudioFormat.MP3,
        sample_rate: Optional[str] = None
    ) -> TTSResult:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            voice_style: Voice style
            audio_format: Output audio format
            sample_rate: Audio sample rate
            
        Returns:
            TTSResult with audio URL and metadata
            
        Raises:
            TTSServiceError: If synthesis fails
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not text or not text.strip():
                raise TTSServiceError("Text cannot be empty")
            
            # Limit text length (Murf has limits)
            if len(text) > 10000:
                logger.warning(f"Text length ({len(text)}) exceeds recommended limit")
                text = text[:10000] + "..."
            
            # Use defaults from config if not provided
            voice_id = voice_id or config.DEFAULT_VOICE_ID
            sample_rate = sample_rate or config.DEFAULT_SAMPLE_RATE
            
            # Prepare request payload
            payload = {
                "text": text.strip(),
                "voiceId": voice_id,
                "style": voice_style.value,
                "format": audio_format.value,
                "sampleRate": sample_rate
            }
            
            logger.info(f"Synthesizing speech: {len(text)} characters with voice {voice_id}")
            logger.debug(f"TTS payload: {payload}")
            
            # Make API request
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            processing_time = time.time() - start_time
            
            # Handle API response
            if response.status_code == 200:
                response_data = response.json()
                audio_url = response_data.get("audioFile")
                
                if not audio_url:
                    raise TTSServiceError("No audio URL in TTS response")
                
                logger.info(f"TTS synthesis successful in {processing_time:.2f}s")
                
                return TTSResult(
                    audio_url=audio_url,
                    duration=response_data.get("duration"),
                    format=audio_format.value,
                    processing_time=processing_time
                )
                
            else:
                error_msg = f"TTS API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('message', response.text)}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                raise TTSServiceError(error_msg)
                
        except TTSServiceError:
            raise
        except requests.exceptions.Timeout:
            error_msg = "TTS request timed out"
            logger.error(error_msg)
            raise TTSServiceError(error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "TTS service connection error"
            logger.error(error_msg)
            raise TTSServiceError(error_msg)
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"TTS service error: {str(e)}"
            logger.error(f"{error_msg} (after {processing_time:.2f}s)")
            raise TTSServiceError(error_msg)
    
    def is_available(self) -> bool:
        """
        Check if TTS service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            return bool(self.api_key and self.api_url)
        except Exception as e:
            logger.error(f"TTS availability check failed: {e}")
            return False
    
    def get_available_voices(self) -> list[str]:
        """
        Get list of available voices.
        
        Returns:
            List of available voice IDs
        """
        # This would ideally come from the API, but Murf doesn't have a voices endpoint
        # So we return commonly available voices
        return [
            "en-US-daniel",
            "en-US-sarah", 
            "en-US-mike",
            "en-US-emma",
            "en-UK-oliver",
            "en-AU-ruby"
        ]
    
    def get_available_styles(self) -> list[str]:
        """
        Get list of available voice styles.
        
        Returns:
            List of available voice styles
        """
        return [style.value for style in VoiceStyle]
    
    async def health_check(self) -> dict:
        """
        Perform health check on TTS service.
        
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
            
            # Test with a short phrase
            try:
                test_result = await self.synthesize_speech(
                    text="Hello", 
                    voice_id=config.DEFAULT_VOICE_ID
                )
                response_time = time.time() - start_time
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "test_synthesis_time": test_result.processing_time,
                    "available_voices": len(self.get_available_voices()),
                    "available_styles": len(self.get_available_styles())
                }
                
            except TTSServiceError as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e), 
                "response_time": time.time() - start_time
            }

# Global TTS service instance
tts_service = TTSService()