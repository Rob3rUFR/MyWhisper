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
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, StreamingResponse
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


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_audio_stream(
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
    SSE streaming transcription endpoint that prevents 504 timeout errors.
    
    Sends periodic progress updates to keep the connection alive,
    then sends the final result when processing completes.
    
    SSE Events:
        - progress: {step, percent, message}
        - result: Final transcription result
        - error: Error message if processing fails
    """
    return await _process_transcription_stream(
        file, language, response_format, diarize, 
        min_speakers, max_speakers, processing_type
    )


async def _process_transcription_stream(
    file: UploadFile,
    language: Optional[str],
    response_format: str,
    diarize: bool,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    processing_type: str = "file"
):
    """
    Process transcription with SSE streaming to prevent proxy timeouts.
    Sends progress updates every few seconds to keep connection alive.
    """
    import asyncio
    from queue import Queue
    from threading import Thread
    
    # Check if already processing
    if processing_state.is_processing:
        status = processing_state.get_status()
        error_msg = (
            f"Un traitement est déjà en cours"
            f" ({status['processing_type']}: {status['current_file']}). "
            f"Veuillez attendre la fin de l'opération."
        )
        
        async def error_stream():
            yield f"event: error\ndata: {json.dumps({'detail': error_msg})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            status_code=503
        )
    
    # Read file content before starting the stream
    file_content = await file.read()
    file_size = len(file_content)
    filename = file.filename
    
    # Validate file
    is_valid, error = validate_audio_file(filename, file_size, settings.MAX_FILE_SIZE)
    if not is_valid:
        async def validation_error():
            yield f"event: error\ndata: {json.dumps({'detail': error})}\n\n"
        return StreamingResponse(
            validation_error(),
            media_type="text/event-stream",
            status_code=400
        )
    
    # Queue for communication between processing thread and SSE stream
    result_queue = Queue()
    
    def run_transcription():
        """Run transcription in a separate thread"""
        temp_path = None
        processing_start_time = time.time()
        
        try:
            # Mark processing as started
            processing_state.start(filename, processing_type)
            
            # Save to temp file
            safe_filename = sanitize_filename(filename)
            temp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}_{safe_filename}"
            
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Processing file (stream): {filename} ({file_size / 1024 / 1024:.1f}MB)")
            
            # Check for cancellation
            if processing_state.cancel_requested:
                result_queue.put(("cancelled", "Traitement annulé par l'utilisateur"))
                return
            
            processing_state.update_progress("transcribing", percent=5)
            
            # Transcribe
            result = transcriber.transcribe(
                str(temp_path),
                language=language,
                vad_filter=True
            )
            
            audio_duration = result.get("duration", 0)
            processing_state.update_progress(
                "transcribing", 
                percent=30 if diarize else 90,
                audio_duration=audio_duration
            )
            
            if processing_state.cancel_requested:
                result_queue.put(("cancelled", "Traitement annulé par l'utilisateur"))
                return
            
            # Diarization if requested
            if diarize:
                if diarizer.is_available:
                    logger.info("Running speaker diarization (stream)...")
                    processing_state.update_progress("diarizing_loading", percent=32)
                    
                    _min = min_speakers if min_speakers else (settings.DIARIZATION_MIN_SPEAKERS or None)
                    _max = max_speakers if max_speakers else (settings.DIARIZATION_MAX_SPEAKERS or None)
                    
                    def on_diarization_progress(step: str, current: int = 0, total: int = 1, sub_percent: float = 0.0):
                        base_percent = 35
                        if step == "loading":
                            percent = base_percent + int(sub_percent * 5)
                        elif step == "processing":
                            percent = 40 + int(sub_percent * 45)
                        elif step == "chunk":
                            chunk_progress = (current / total) if total > 0 else 0
                            percent = 40 + int(chunk_progress * 45)
                        elif step == "merging":
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
                    
                    if processing_state.cancel_requested:
                        result_queue.put(("cancelled", "Traitement annulé par l'utilisateur"))
                        return
                    
                    processing_state.update_progress("merging", percent=90)
                    
                    result["segments"] = diarizer.merge_with_transcription(
                        result["segments"],
                        timeline
                    )
                    
                    result["text"] = segments_to_text(
                        result["segments"],
                        include_speakers=True
                    )
                else:
                    logger.warning("Diarization requested but not available")
            
            processing_state.update_progress("finalizing", percent=95)
            
            # Format response
            if response_format == "text":
                result_content = result["text"]
            elif response_format == "srt":
                result_content = segments_to_srt(result["segments"])
            elif response_format == "vtt":
                result_content = segments_to_vtt(result["segments"])
            else:
                result_content = result
            
            # Save to history
            if processing_type == "file":
                try:
                    speakers_count = 0
                    if diarize and result.get("segments"):
                        speakers = set(s.get("speaker") for s in result["segments"] if s.get("speaker"))
                        speakers_count = len(speakers)
                    
                    processing_duration = round(time.time() - processing_start_time, 2)
                    
                    history.save_transcription(
                        filename=filename,
                        file_size=file_size,
                        audio_duration=result.get("duration", 0),
                        language=result.get("language", "unknown"),
                        format=response_format,
                        diarization=diarize,
                        speakers_count=speakers_count,
                        segments_count=len(result.get("segments", [])),
                        result_text=result_content if isinstance(result_content, str) else json.dumps(result_content, ensure_ascii=False, indent=2),
                        result_json=result,
                        processing_duration=processing_duration
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to history: {e}")
            
            result_queue.put(("success", result_content, response_format))
            
        except Exception as e:
            logger.error(f"Transcription error (stream): {e}")
            result_queue.put(("error", str(e)))
        finally:
            processing_state.stop()
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
    
    async def event_stream():
        """Generate SSE events with progress updates"""
        # Start transcription in background thread
        thread = Thread(target=run_transcription)
        thread.start()
        
        last_status = None
        
        # Send progress updates while processing
        while thread.is_alive():
            # Check for result
            if not result_queue.empty():
                break
            
            # Get current status
            status = processing_state.get_status()
            
            # Send progress update (only if changed)
            progress_data = {
                "step": status["current_step"],
                "percent": status["progress_percent"],
                "audio_duration": status["audio_duration"],
                "current_chunk": status["current_chunk"],
                "total_chunks": status["total_chunks"]
            }
            
            if progress_data != last_status:
                yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"
                last_status = progress_data
            
            # Wait before next update (keeps connection alive)
            await asyncio.sleep(1)
        
        # Wait for thread to complete
        thread.join(timeout=5)
        
        # Get final result
        if not result_queue.empty():
            result = result_queue.get()
            
            if result[0] == "success":
                result_content = result[1]
                fmt = result[2]
                
                # Send final progress
                yield f"event: progress\ndata: {json.dumps({'step': 'complete', 'percent': 100})}\n\n"
                
                # Send result based on format
                if isinstance(result_content, str):
                    yield f"event: result\ndata: {json.dumps({'format': fmt, 'content': result_content})}\n\n"
                else:
                    yield f"event: result\ndata: {json.dumps({'format': fmt, 'content': result_content})}\n\n"
                    
            elif result[0] == "cancelled":
                yield f"event: cancelled\ndata: {json.dumps({'message': result[1]})}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'detail': result[1]})}\n\n"
        else:
            yield f"event: error\ndata: {json.dumps({'detail': 'Processing failed unexpectedly'})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


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
    
    Uses asyncio.to_thread() to run Whisper in a separate thread,
    keeping the server responsive during transcription.
    
    1. Check if another transcription is in progress
    2. Validate file (format, size)
    3. Save temporarily in /uploads
    4. Run Whisper transcription (in thread)
    5. Run diarization if requested (in thread)
    6. Format response
    7. Cleanup temporary file
    """
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
    
    # Validate file first (before acquiring lock)
    file_content = await file.read()
    file_size = len(file_content)
    filename = file.filename
    
    is_valid, error = validate_audio_file(filename, file_size, settings.MAX_FILE_SIZE)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Acquire the lock for processing
    async with processing_state.lock:
        # Double-check after acquiring lock
        if processing_state.is_processing:
            status = processing_state.get_status()
            error_msg = (
                f"Un traitement est déjà en cours"
                f" ({status['processing_type']}: {status['current_file']}). "
                f"Veuillez attendre la fin de l'opération."
            )
            raise HTTPException(status_code=503, detail=error_msg)
        
        # Mark processing as started
        processing_state.start(filename, processing_type)
    
    # Now run the heavy processing in a separate thread
    # This allows the server to remain responsive
    def run_blocking_transcription():
        """
        Synchronous function that runs in a separate thread.
        This prevents blocking the asyncio event loop.
        """
        temp_path = None
        processing_start_time = time.time()
        
        try:
            # Save to temp file
            safe_filename = sanitize_filename(filename)
            temp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}_{safe_filename}"
            
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Processing file: {filename} ({file_size / 1024 / 1024:.1f}MB)")
            
            # Check for cancellation before transcription
            if processing_state.cancel_requested:
                logger.info("Processing cancelled before transcription")
                return {"error": "cancelled", "message": "Traitement annulé par l'utilisateur"}
            
            # Update progress: transcription starting
            processing_state.update_progress("transcribing", percent=5)
            
            # Transcribe (blocking call)
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
            
            # Check for cancellation after transcription
            if processing_state.cancel_requested:
                logger.info("Processing cancelled after transcription")
                return {"error": "cancelled", "message": "Traitement annulé par l'utilisateur"}
            
            # Diarization if requested
            if diarize:
                if diarizer.is_available:
                    logger.info("Running speaker diarization...")
                    processing_state.update_progress("diarizing_loading", percent=32)
                    
                    _min = min_speakers if min_speakers else (settings.DIARIZATION_MIN_SPEAKERS or None)
                    _max = max_speakers if max_speakers else (settings.DIARIZATION_MAX_SPEAKERS or None)
                    
                    def on_diarization_progress(step: str, current: int = 0, total: int = 1, sub_percent: float = 0.0):
                        base_percent = 35
                        if step == "loading":
                            percent = base_percent + int(sub_percent * 5)
                        elif step == "processing":
                            percent = 40 + int(sub_percent * 45)
                        elif step == "chunk":
                            chunk_progress = (current / total) if total > 0 else 0
                            percent = 40 + int(chunk_progress * 45)
                        elif step == "merging":
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
                        return {"error": "cancelled", "message": "Traitement annulé par l'utilisateur"}
                    
                    # Merge transcription with diarization
                    processing_state.update_progress("merging", percent=90)
                    
                    result["segments"] = diarizer.merge_with_transcription(
                        result["segments"],
                        timeline
                    )
                    
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
                result_content = result
            
            # Save to history (only for file transcriptions, not dictation)
            if processing_type == "file":
                try:
                    speakers_count = 0
                    if diarize and result.get("segments"):
                        speakers = set(s.get("speaker") for s in result["segments"] if s.get("speaker"))
                        speakers_count = len(speakers)
                    
                    processing_duration = round(time.time() - processing_start_time, 2)
                    
                    history.save_transcription(
                        filename=filename,
                        file_size=file_size,
                        audio_duration=result.get("duration", 0),
                        language=result.get("language", "unknown"),
                        format=response_format,
                        diarization=diarize,
                        speakers_count=speakers_count,
                        segments_count=len(result.get("segments", [])),
                        result_text=result_content if isinstance(result_content, str) else json.dumps(result_content, ensure_ascii=False, indent=2),
                        result_json=result,
                        processing_duration=processing_duration
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to history: {e}")
            
            return {
                "success": True,
                "result": result,
                "result_content": result_content,
                "format": response_format
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": "exception", "message": str(e)}
        finally:
            processing_state.stop()
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
    
    # Run the blocking transcription in a thread pool
    # This keeps the server responsive during processing
    result = await asyncio.to_thread(run_blocking_transcription)
    
    # Handle the result
    if "error" in result:
        if result["error"] == "cancelled":
            raise HTTPException(status_code=499, detail=result["message"])
        else:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {result['message']}")
    
    # Return appropriate response based on format
    result_content = result["result_content"]
    fmt = result["format"]
    
    if fmt == "text":
        return PlainTextResponse(content=result_content)
    elif fmt == "srt":
        return PlainTextResponse(content=result_content, media_type="text/plain")
    elif fmt == "vtt":
        return PlainTextResponse(content=result_content, media_type="text/vtt")
    else:
        return JSONResponse(content=result_content)


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
