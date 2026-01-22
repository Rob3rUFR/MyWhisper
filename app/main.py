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
import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import httpx

from app.config import settings
from app.transcription import transcriber
from app.diarization import diarizer
from app.history import history
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


@dataclass
class ProcessingState:
    """
    Global state to track if a transcription is in progress.
    Prevents concurrent processing that could crash the GPU.
    """
    is_processing: bool = False
    current_file: Optional[str] = None
    started_at: Optional[datetime] = None
    processing_type: Optional[str] = None  # 'file' or 'dictation'
    cancel_requested: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Progress tracking
    current_step: str = "idle"  # 'uploading', 'transcribing', 'diarizing', 'finalizing'
    progress_percent: int = 0
    total_chunks: int = 0
    current_chunk: int = 0
    audio_duration: float = 0.0
    
    def start(self, filename: str, processing_type: str = "file"):
        """Mark processing as started"""
        self.is_processing = True
        self.current_file = filename
        self.started_at = datetime.now()
        self.processing_type = processing_type
        self.cancel_requested = False
        self.current_step = "uploading"
        self.progress_percent = 0
        self.total_chunks = 0
        self.current_chunk = 0
        self.audio_duration = 0.0
    
    def stop(self):
        """Mark processing as completed"""
        self.is_processing = False
        self.current_file = None
        self.started_at = None
        self.processing_type = None
        self.cancel_requested = False
        self.current_step = "idle"
        self.progress_percent = 0
        self.total_chunks = 0
        self.current_chunk = 0
        self.audio_duration = 0.0
    
    def update_progress(self, step: str, percent: int = None, 
                       current_chunk: int = None, total_chunks: int = None,
                       audio_duration: float = None):
        """Update progress information"""
        self.current_step = step
        if percent is not None:
            self.progress_percent = percent
        if current_chunk is not None:
            self.current_chunk = current_chunk
        if total_chunks is not None:
            self.total_chunks = total_chunks
        if audio_duration is not None:
            self.audio_duration = audio_duration
    
    def request_cancel(self) -> bool:
        """Request cancellation of current processing"""
        if self.is_processing:
            self.cancel_requested = True
            return True
        return False
    
    def get_status(self) -> dict:
        """Get current processing status"""
        elapsed = None
        if self.started_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
        
        return {
            "is_processing": self.is_processing,
            "current_file": self.current_file,
            "processing_type": self.processing_type,
            "elapsed_seconds": elapsed,
            "cancel_requested": self.cancel_requested,
            "current_step": self.current_step,
            "progress_percent": self.progress_percent,
            "total_chunks": self.total_chunks,
            "current_chunk": self.current_chunk,
            "audio_duration": self.audio_duration
        }


# Global processing state
processing_state = ProcessingState()


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


@app.get("/status")
async def get_processing_status():
    """
    Get current processing status.
    Used by frontend to check if transcription/dictation can be started.
    """
    return processing_state.get_status()


@app.post("/cancel")
async def cancel_processing():
    """
    Request cancellation of current processing.
    Note: Cancellation is not immediate as Whisper processing is blocking.
    The processing will stop at the next opportunity.
    """
    if processing_state.request_cancel():
        logger.info(f"Cancellation requested for: {processing_state.current_file}")
        return {
            "success": True,
            "message": "Annulation demandée. Le traitement s'arrêtera dès que possible."
        }
    return {
        "success": False,
        "message": "Aucun traitement en cours à annuler."
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
    max_speakers: Optional[int] = Form(None),
    processing_type: str = Form("file")
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
        processing_type: Type of processing - 'file' (saved to history) or 'dictation' (not saved)
        
    Returns:
        Transcription in requested format
    """
    return await _process_transcription(file, language, response_format, diarize, min_speakers, max_speakers, processing_type)


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
    max_speakers: Optional[int] = None,
    processing_type: str = "file"
) -> JSONResponse:
    """
    Process transcription request.
    
    1. Check if another transcription is in progress
    2. Validate file (format, size)
    3. Save temporarily in /uploads
    4. Run Whisper transcription
    5. Run diarization if requested
    6. Format response
    7. Cleanup temporary file
    """
    temp_path = None
    processing_start_time = time.time()  # Track processing duration
    
    # Try to acquire the lock without blocking to check status
    if processing_state.is_processing:
        status = processing_state.get_status()
        error_msg = (
            f"Un traitement est déjà en cours"
            f" ({status['processing_type']}: {status['current_file']}). "
            f"Veuillez attendre la fin de l'opération."
        )
        logger.warning(f"Rejected request: {error_msg}")
        raise HTTPException(status_code=503, detail=error_msg)
    
    # Acquire the lock for processing
    async with processing_state.lock:
        try:
            # Double-check after acquiring lock
            if processing_state.is_processing:
                status = processing_state.get_status()
                error_msg = (
                    f"Un traitement est déjà en cours"
                    f" ({status['processing_type']}: {status['current_file']}). "
                    f"Veuillez attendre la fin de l'opération."
                )
                raise HTTPException(status_code=503, detail=error_msg)
            
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
            
            # Mark processing as started
            processing_state.start(file.filename, processing_type)
            
            # Save to temp file
            safe_filename = sanitize_filename(file.filename)
            temp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}_{safe_filename}"
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"Processing file: {file.filename} ({file_size / 1024 / 1024:.1f}MB)")
            
            # Check for cancellation before transcription
            if processing_state.cancel_requested:
                logger.info("Processing cancelled before transcription")
                raise HTTPException(status_code=499, detail="Traitement annulé par l'utilisateur")
            
            # Update progress: transcription starting
            # Progress distribution with diarization: transcription 5-30%, diarization 30-90%, merge 90-95%, finalize 95-100%
            # Progress distribution without diarization: transcription 5-90%, finalize 90-100%
            processing_state.update_progress("transcribing", percent=5)
            
            # Transcribe
            result = transcriber.transcribe(
                str(temp_path),
                language=language,
                vad_filter=True
            )
            
            # Update progress: transcription complete
            audio_duration = result.get("duration", 0)
            processing_state.update_progress(
                "transcribing", 
                percent=30 if diarize else 90,
                audio_duration=audio_duration
            )
            
            # Check for cancellation after transcription, before diarization
            if processing_state.cancel_requested:
                logger.info("Processing cancelled after transcription")
                raise HTTPException(status_code=499, detail="Traitement annulé par l'utilisateur")
            
            # Diarization if requested
            if diarize:
                if diarizer.is_available:
                    logger.info("Running speaker diarization...")
                    processing_state.update_progress("diarizing_loading", percent=32)
                    
                    # Use provided params or fall back to config defaults
                    _min = min_speakers if min_speakers else (settings.DIARIZATION_MIN_SPEAKERS or None)
                    _max = max_speakers if max_speakers else (settings.DIARIZATION_MAX_SPEAKERS or None)
                    
                    # Progress callback for diarization steps
                    # Diarization goes from 35% to 88% of total progress
                    def on_diarization_progress(step: str, current: int = 0, total: int = 1, sub_percent: float = 0.0):
                        """
                        Enhanced progress callback that handles both chunk-based and sub-step progress.
                        
                        Args:
                            step: Current step name ('loading', 'processing', 'chunk', 'merging')
                            current: Current item number (e.g., chunk number)
                            total: Total items (e.g., total chunks)
                            sub_percent: Sub-progress within current step (0.0 to 1.0)
                        """
                        base_percent = 35  # Where diarization starts
                        range_percent = 53  # Range for diarization (35 to 88)
                        
                        if step == "loading":
                            # Loading pipeline: 35-40%
                            percent = base_percent + int(sub_percent * 5)
                        elif step == "processing":
                            # Processing (for short audio): 40-85%
                            percent = 40 + int(sub_percent * 45)
                        elif step == "chunk":
                            # Chunk-based progress: 40-85%
                            chunk_progress = (current / total) if total > 0 else 0
                            percent = 40 + int(chunk_progress * 45)
                        elif step == "merging":
                            # Merging speakers: 85-88%
                            percent = 85 + int(sub_percent * 3)
                        else:
                            percent = base_percent
                        
                        processing_state.update_progress(
                            f"diarizing_{step}",
                            percent=min(percent, 88),
                            current_chunk=current,
                            total_chunks=total
                        )
                    
                    timeline = diarizer.diarize(
                        str(temp_path),
                        min_speakers=_min,
                        max_speakers=_max,
                        progress_callback=on_diarization_progress
                    )
                    
                    # Check for cancellation after diarization
                    if processing_state.cancel_requested:
                        logger.info("Processing cancelled after diarization")
                        raise HTTPException(status_code=499, detail="Traitement annulé par l'utilisateur")
                    
                    # Merge transcription with diarization
                    processing_state.update_progress("merging", percent=90)
                    
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
            
            # Update progress: finalizing
            processing_state.update_progress("finalizing", percent=95)
            
            # Format response based on requested format
            if response_format == "text":
                result_content = result["text"]
            elif response_format == "srt":
                result_content = segments_to_srt(result["segments"])
            elif response_format == "vtt":
                result_content = segments_to_vtt(result["segments"])
            else:
                # JSON format
                result_content = result
            
            # Save to history (only for file transcriptions, not dictation)
            if processing_type == "file":
                try:
                    # Count speakers if diarization was done
                    speakers_count = 0
                    if diarize and result.get("segments"):
                        speakers = set(s.get("speaker") for s in result["segments"] if s.get("speaker"))
                        speakers_count = len(speakers)
                    
                    # Calculate processing duration
                    processing_duration = round(time.time() - processing_start_time, 2)
                    
                    # Save with formatted result
                    history.save_transcription(
                        filename=file.filename,
                        file_size=file_size,
                        audio_duration=result.get("duration", 0),
                        language=result.get("language", "unknown"),
                        format=response_format,
                        diarization=diarize,
                        speakers_count=speakers_count,
                        segments_count=len(result.get("segments", [])),
                        result_text=result_content if isinstance(result_content, str) else json.dumps(result_content, ensure_ascii=False, indent=2),
                        result_json=result,  # Always save full JSON for re-formatting
                        processing_duration=processing_duration
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to history: {e}")
            
            # Return response
            if response_format == "text":
                return PlainTextResponse(content=result_content)
            elif response_format == "srt":
                return PlainTextResponse(content=result_content, media_type="text/plain")
            elif response_format == "vtt":
                return PlainTextResponse(content=result_content, media_type="text/vtt")
            else:
                return JSONResponse(content=result_content)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        finally:
            # Mark processing as complete
            processing_state.stop()
            
            # Cleanup temp file
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")


# ============================================
# History Endpoints
# ============================================

@app.get("/history")
async def list_history(limit: int = 50, offset: int = 0):
    """
    List transcription history with pagination.
    
    Args:
        limit: Maximum number of records (default 50)
        offset: Number of records to skip (for pagination)
        
    Returns:
        List of transcription records
    """
    transcriptions = history.list_transcriptions(limit=limit, offset=offset)
    total = history.get_total_count()
    
    return {
        "transcriptions": transcriptions,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/history/{record_id}")
async def get_history_record(record_id: int):
    """
    Get a specific transcription from history.
    
    Args:
        record_id: The transcription ID
        
    Returns:
        Full transcription record including result
    """
    record = history.get_transcription(record_id)
    
    if not record:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    return record


@app.get("/history/{record_id}/download")
async def download_history_record(record_id: int, format: Optional[str] = None):
    """
    Download a transcription result in the specified format.
    
    Args:
        record_id: The transcription ID
        format: Output format (text, json, srt, vtt). Defaults to original format.
        
    Returns:
        Transcription content in requested format
    """
    record = history.get_transcription(record_id)
    
    if not record:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # Use original format if not specified
    output_format = format or record.get("format", "text")
    
    # Get the full JSON result for re-formatting
    result_json = record.get("result_json")
    
    if output_format == "text":
        if result_json and "text" in result_json:
            content = result_json["text"]
        else:
            content = record.get("result_text", "")
        return PlainTextResponse(content=content)
    
    elif output_format == "srt":
        if result_json and "segments" in result_json:
            content = segments_to_srt(result_json["segments"])
        else:
            content = record.get("result_text", "")
        return PlainTextResponse(content=content, media_type="text/plain")
    
    elif output_format == "vtt":
        if result_json and "segments" in result_json:
            content = segments_to_vtt(result_json["segments"])
        else:
            content = record.get("result_text", "")
        return PlainTextResponse(content=content, media_type="text/vtt")
    
    else:
        # JSON format
        if result_json:
            return JSONResponse(content=result_json)
        else:
            return PlainTextResponse(content=record.get("result_text", ""))


@app.delete("/history/{record_id}")
async def delete_history_record(record_id: int):
    """
    Delete a transcription from history.
    
    Args:
        record_id: The transcription ID
        
    Returns:
        Success message
    """
    deleted = history.delete_transcription(record_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    return {"success": True, "message": "Transcription deleted"}


# ============================================
# Settings Endpoints
# ============================================

class RetentionDaysRequest(BaseModel):
    days: int


@app.get("/settings")
async def get_settings():
    """
    Get user-configurable settings.
    
    Returns:
        Current settings including history retention days
    """
    return {
        "retention_days": history.get_retention_days()
    }


@app.post("/settings/retention")
async def set_retention_days(request: RetentionDaysRequest):
    """
    Set the number of days to retain transcription history.
    
    Args:
        days: Number of days (1-365)
        
    Returns:
        Success status and current value
    """
    success = history.set_retention_days(request.days)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")
    
    return {
        "success": True,
        "retention_days": history.get_retention_days()
    }


# ============================================
# Ollama Proxy Endpoints
# ============================================

@app.get("/ollama/status")
async def ollama_status():
    """
    Check if Ollama is configured and reachable.
    Returns connection status and available model.
    """
    if not settings.OLLAMA_URL:
        return {
            "configured": False,
            "connected": False,
            "url": None,
            "model": None,
            "models": [],
            "message": "Ollama non configuré (OLLAMA_URL vide)"
        }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                
                # Use configured model or first available
                selected_model = settings.OLLAMA_MODEL if settings.OLLAMA_MODEL else (models[0] if models else None)
                
                return {
                    "configured": True,
                    "connected": True,
                    "url": settings.OLLAMA_URL,
                    "model": selected_model,
                    "models": models,
                    "message": f"Connecté - {len(models)} modèle(s) disponible(s)"
                }
            else:
                return {
                    "configured": True,
                    "connected": False,
                    "url": settings.OLLAMA_URL,
                    "model": None,
                    "models": [],
                    "message": f"Erreur HTTP {response.status_code}"
                }
    except httpx.ConnectError:
        return {
            "configured": True,
            "connected": False,
            "url": settings.OLLAMA_URL,
            "model": None,
            "models": [],
            "message": "Impossible de se connecter à Ollama"
        }
    except Exception as e:
        return {
            "configured": True,
            "connected": False,
            "url": settings.OLLAMA_URL,
            "model": None,
            "models": [],
            "message": f"Erreur: {str(e)}"
        }


@app.get("/ollama/models")
async def ollama_models():
    """
    List available Ollama models.
    """
    if not settings.OLLAMA_URL:
        raise HTTPException(status_code=503, detail="Ollama non configuré")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"models": models}
            else:
                raise HTTPException(status_code=response.status_code, detail="Erreur Ollama")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Impossible de se connecter à Ollama")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ollama/generate")
async def ollama_generate(request: Request):
    """
    Generate text using Ollama.
    Proxies the request to Ollama server.
    
    Body:
        model: Model name (optional, uses OLLAMA_MODEL if not provided)
        prompt: The prompt to send
    """
    if not settings.OLLAMA_URL:
        raise HTTPException(status_code=503, detail="Ollama non configuré")
    
    try:
        body = await request.json()
        
        # Use configured model if not specified
        if not body.get("model") and settings.OLLAMA_MODEL:
            body["model"] = settings.OLLAMA_MODEL
        
        if not body.get("model"):
            raise HTTPException(status_code=400, detail="Aucun modèle spécifié")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json=body
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="Erreur Ollama")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Impossible de se connecter à Ollama")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ollama generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
