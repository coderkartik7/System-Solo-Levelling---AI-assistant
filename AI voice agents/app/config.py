"""Configuration management for AI Voice Agent."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # API Keys (Required)
    MURF_API_KEY: Optional[str] = os.getenv("MURF_API_KEY")
    ASSEMBLY_API_KEY: Optional[str] = os.getenv("ASSEMBLY_API_KEY") 
    GEMINI_API_KEY: Optional[str] = os.getenv("G_API_KEY")
    
    # API URLs
    MURF_API_URL: str = os.getenv("MURF_API_URL", "https://api.murf.ai/v1/speech/generate")
    
    # App Settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # File Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # MB to bytes
    ALLOWED_AUDIO_EXTENSIONS: set = {".mp3", ".wav", ".webm", ".m4a", ".ogg"}
    
    # Session Settings
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # seconds
    MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # TTS Settings
    DEFAULT_VOICE_ID: str = os.getenv("DEFAULT_VOICE_ID", "en-US-daniel")
    DEFAULT_VOICE_STYLE: str = os.getenv("DEFAULT_VOICE_STYLE", "Conversational")
    DEFAULT_AUDIO_FORMAT: str = os.getenv("DEFAULT_AUDIO_FORMAT", "MP3")
    DEFAULT_SAMPLE_RATE: str = os.getenv("DEFAULT_SAMPLE_RATE", "8000.0")
    
    # LLM Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        missing_keys = []
        
        if not cls.MURF_API_KEY:
            missing_keys.append("MURF_API_KEY")
        if not cls.ASSEMBLY_API_KEY:
            missing_keys.append("ASSEMBLY_API_KEY")
        if not cls.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
            
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories."""
        cls.UPLOADS_DIR.mkdir(exist_ok=True)
        
    @classmethod
    def setup_logging(cls) -> None:
        """Setup application logging."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log') if not cls.DEBUG else logging.NullHandler()
            ]
        )

# Global config instance
config = Config()