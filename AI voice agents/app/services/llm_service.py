"""Large Language Model service using Google Gemini."""

import logging
import time
from typing import List, Optional
from google import genai
from google.genai import types
from typing import Optional
from ..config import config
from ..models.schemas import ChatMessage, LLMResult

logger = logging.getLogger(__name__)

class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""
    pass

class LLMService:
    """Large Language Model service wrapper."""
    
    def __init__(self):
        """Initialize LLM service."""
        if not config.GEMINI_API_KEY:
            raise LLMServiceError("Gemini API key not configured")
        
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        logger.info("LLM Service initialized with Google Gemini")
    
    async def generate_response(
        self,
        prompt: str,
        chat_history: Optional[List[ChatMessage]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        disable_thinking: bool = False
    ) -> LLMResult:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            chat_history: Previous chat messages for context
            model: Model to use
            temperature: Response randomness (0.0-2.0)
            max_tokens: Maximum response tokens
            disable_thinking: Disable thinking mode
            
        Returns:
            LLMResult with generated response and metadata
            
        Raises:
            LLMServiceError: If generation fails
        """
        start_time = time.time()
        
        try:
            # Use defaults from config if not provided
            model = model or config.DEFAULT_MODEL
            temperature = temperature or config.DEFAULT_TEMPERATURE
            max_tokens = max_tokens or config.DEFAULT_MAX_TOKENS
            
            # Validate input
            if not prompt or not prompt.strip():
                raise LLMServiceError("Prompt cannot be empty")
            
            # Build context from chat history
            context_str = self._build_context(chat_history) if chat_history else ""
            full_prompt = f"{context_str}\nUser: {prompt}" if context_str else prompt
            
            logger.info(f"Generating response with model {model}")
            logger.debug(f"Full prompt length: {len(full_prompt)} characters")
            
            # Configure generation parameters
            generation_config = None
            thinking_config = None
            
            if disable_thinking:
                thinking_config = types.ThinkingConfig(thinking_budget=0)
            
            # Make API request
            response = self.client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_config
                ) if thinking_config else types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            processing_time = time.time() - start_time
            
            # Validate response
            if not response.candidates:
                raise LLMServiceError("No response candidates received from LLM")
            
            if not response.text:
                raise LLMServiceError("Empty response received from LLM")
            
            response_text = response.text.strip()
            
            logger.info(
                f"LLM response generated: {len(response_text)} characters "
                f"in {processing_time:.2f}s"
            )
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, 'total_token_count', None)
            
            return LLMResult(
                response=response_text,
                model_used=model,
                tokens_used=tokens_used,
                processing_time=processing_time
            )
            
        except LLMServiceError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"LLM service error: {str(e)}"
            logger.error(f"{error_msg} (after {processing_time:.2f}s)")
            raise LLMServiceError(error_msg)
    
    def _build_context(self, chat_history: List[ChatMessage]) -> str:
        """
        Build context string from chat history.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            Formatted context string
        """
        if not chat_history:
            return ""
        
        # Limit context to recent messages to avoid token limits
        recent_messages = chat_history[-config.MAX_CHAT_HISTORY:]
        
        context_parts = []
        for message in recent_messages:
            if message.role.value == "user":
                context_parts.append(f"User: {message.content}")
            else:
                context_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(context_parts)
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            return bool(config.GEMINI_API_KEY and self.client)
        except Exception as e:
            logger.error(f"LLM availability check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        return [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    
    async def health_check(self) -> dict:
        """
        Perform health check on LLM service.
        
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
            
            # Test with a simple prompt
            try:
                test_result = await self.generate_response(
                    prompt="Hello, respond with 'OK'",
                    model=config.DEFAULT_MODEL
                )
                response_time = time.time() - start_time
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "test_generation_time": test_result.processing_time,
                    "available_models": len(self.get_available_models()),
                    "default_model": config.DEFAULT_MODEL
                }
                
            except LLMServiceError as e:
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

# Global LLM service instance
llm_service = LLMService()