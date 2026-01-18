"""
FastAPI application for Whisper STT service
OpenAI-compatible API with web interface
"""
# IMPORTANT: Import patches FIRST before any other imports
# This patches torch.load and torchaudio for PyTorch 2.6+ compatibility
import app.patches  # noqa: F401

import os
import uuid
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from app.config import settings
from app.transcription import transcriber
from app.diarization import diarizer
from app.utils import (
    validate_audio_file,
    sanitize_filename,
    segments_to_srt,
    segments_to_vtt,
    segments_to_text
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown"""
    # Startup
    logger.info("=" * 50)
    logger.info("🎤 Whisper STT Service Starting...")
    logger.info("=" * 50)
    
    # Create directories
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # Pre-load Whisper model
    logger.info("Loading Whisper model (this may take a moment)...")
    try:
        _ = transcriber.model
        logger.info("✅ Whisper model ready")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
    
    # Check diarization availability
    if diarizer.is_available:
        logger.info("✅ Diarization available")
    else:
        logger.warning("⚠️ Diarization unavailable (check HF_TOKEN)")
    
    logger.info("=" * 50)
    logger.info("🚀 Service ready!")
    logger.info(f"📡 API: http://{settings.HOST}:{settings.PORT}")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Whisper STT Service...")


# Create FastAPI application
app = FastAPI(
    title="Whisper STT API",
    description="Speech-to-Text service using Faster Whisper with speaker diarization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    index_path = static_path / "index.html"
    if index_path.exists():
        async with aiofiles.open(index_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>Whisper STT</h1><p>Static files not found</p>")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and Open WebUI integration.
    Returns GPU status and model information.
    """
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None
    
    if gpu_available:
        gpu_memory = {
            "total": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "allocated": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "cached": round(torch.cuda.memory_reserved(0) / 1024**3, 2)
        }
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "model": transcriber.get_model_info(),
        "diarization": diarizer.get_info()
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """
    List available models (OpenAI-compatible endpoint).
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-large-v3",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
                "permission": [],
                "root": "whisper-large-v3",
                "parent": None
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio_openai(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    diarize: bool = Form(False),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """
    OpenAI-compatible transcription endpoint.
    
    Args:
        file: Audio file to transcribe
        model: Model name (ignored, always uses configured model)
        language: Language code (e.g., 'fr', 'en'). None for auto-detection
        response_format: Output format - json, text, srt, vtt
        diarize: Enable speaker diarization
        min_speakers: Minimum number of speakers (optional, speeds up diarization)
        max_speakers: Maximum number of speakers (optional, speeds up diarization)
        
    Returns:
        Transcription in requested format
    """
    return await _process_transcription(file, language, response_format, diarize, min_speakers, max_speakers)


@app.post("/transcribe")
async def simple_transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """
    Simplified transcription endpoint, always returns JSON.
    """
    return await _process_transcription(file, language, "json", diarize, min_speakers, max_speakers)


async def _process_transcription(
    file: UploadFile,
    language: Optional[str],
    response_format: str,
    diarize: bool,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> JSONResponse:
    """
    Process transcription request.
    
    1. Validate file (format, size)
    2. Save temporarily in /uploads
    3. Run Whisper transcription
    4. Run diarization if requested
    5. Format response
    6. Cleanup temporary file
    """
    temp_path = None
    
    try:
        # Validate file
        file_content = await file.read()
        file_size = len(file_content)
        
        is_valid, error = validate_audio_file(
            file.filename,
            file_size,
            settings.MAX_FILE_SIZE
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Save to temp file
        safe_filename = sanitize_filename(file.filename)
        temp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}_{safe_filename}"
        
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"Processing file: {file.filename} ({file_size / 1024 / 1024:.1f}MB)")
        
        # Transcribe
        result = transcriber.transcribe(
            str(temp_path),
            language=language,
            vad_filter=True
        )
        
        # Diarization if requested
        if diarize:
            if diarizer.is_available:
                logger.info("Running speaker diarization...")
                
                # Use provided params or fall back to config defaults
                _min = min_speakers if min_speakers else (settings.DIARIZATION_MIN_SPEAKERS or None)
                _max = max_speakers if max_speakers else (settings.DIARIZATION_MAX_SPEAKERS or None)
                
                timeline = diarizer.diarize(
                    str(temp_path),
                    min_speakers=_min,
                    max_speakers=_max
                )
                result["segments"] = diarizer.merge_with_transcription(
                    result["segments"],
                    timeline
                )
                
                # Update full text with speaker labels
                result["text"] = segments_to_text(
                    result["segments"],
                    include_speakers=True
                )
            else:
                logger.warning("Diarization requested but not available")
        
        # Format response
        if response_format == "text":
            return PlainTextResponse(content=result["text"])
        elif response_format == "srt":
            srt_content = segments_to_srt(result["segments"])
            return PlainTextResponse(content=srt_content, media_type="text/plain")
        elif response_format == "vtt":
            vtt_content = segments_to_vtt(result["segments"])
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")
        else:
            # JSON format (default)
            return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
