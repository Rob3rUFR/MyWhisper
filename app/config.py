"""
Configuration management for Whisper STT service
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Model Configuration
    WHISPER_MODEL: str = "large-v3"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"
    NUM_WORKERS: int = 4
    
    # Diarization
    ENABLE_DIARIZATION: bool = True
    HF_TOKEN: str = ""  # Required if diarization enabled
    DIARIZATION_MIN_SPEAKERS: int = 0  # 0 = auto-detect
    DIARIZATION_MAX_SPEAKERS: int = 0  # 0 = auto-detect
    
    # Limits
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_FORMATS: List[str] = ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
    TRANSCRIPTION_TIMEOUT: int = 300  # 5 minutes
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MODEL_DIR: str = "./models"
    
    # Ollama (LLM post-processing)
    OLLAMA_URL: str = ""  # e.g. http://localhost:11434 or http://ollama:11434
    OLLAMA_MODEL: str = ""  # e.g. llama3.2, mistral, etc.
    
    # History settings
    HISTORY_RETENTION_DAYS: int = 90  # Default: 90 days
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()
