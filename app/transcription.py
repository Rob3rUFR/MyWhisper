"""
Whisper transcription engine using faster-whisper
"""
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import settings

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Transcription engine using Faster Whisper for optimized inference.
    Designed for RTX 3090 with CUDA acceleration.
    """
    
    _instance: Optional['WhisperTranscriber'] = None
    _model: Optional[WhisperModel] = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the transcriber (only loads model once)"""
        if self._model is None:
            self._load_model()
    
    def _load_model(self) -> None:
        """
        Load Faster Whisper model with optimized settings for RTX 3090.
        
        Configuration:
        - device: cuda for GPU acceleration
        - compute_type: float16 for optimal speed/quality ratio on 3090
        - num_workers: 4 for parallel processing
        """
        logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
        logger.info(f"Device: {settings.DEVICE}, Compute type: {settings.COMPUTE_TYPE}")
        
        model_path = Path(settings.MODEL_DIR) / "whisper"
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self._model = WhisperModel(
                model_size_or_path=settings.WHISPER_MODEL,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                num_workers=settings.NUM_WORKERS,
                download_root=str(model_path)
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU if GPU fails
            if settings.DEVICE == "cuda":
                logger.warning("Falling back to CPU...")
                self._model = WhisperModel(
                    model_size_or_path=settings.WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                    num_workers=settings.NUM_WORKERS,
                    download_root=str(model_path)
                )
                logger.info("Whisper model loaded on CPU")
            else:
                raise
    
    @property
    def model(self) -> WhisperModel:
        """Get the loaded model"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        vad_filter: bool = True,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'fr', 'en'). None for auto-detection
            task: 'transcribe' or 'translate'
            vad_filter: Apply Voice Activity Detection to filter silences
            word_timestamps: Include word-level timestamps
            
        Returns:
            Dictionary with transcription results:
            {
                "text": str,
                "language": str,
                "duration": float,
                "segments": List[Dict]
            }
        """
        logger.info(f"Starting transcription: {audio_path}")
        logger.info(f"Language: {language or 'auto-detect'}, VAD: {vad_filter}")
        
        try:
            segments_gen, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                beam_size=5,
                best_of=5,
                patience=1.0,
                condition_on_previous_text=True
            )
            
            # Process segments
            segments = []
            full_text = []
            
            for segment in segments_gen:
                seg_dict = {
                    "id": segment.id,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip()
                }
                
                if word_timestamps and segment.words:
                    seg_dict["words"] = [
                        {
                            "word": w.word,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 3)
                        }
                        for w in segment.words
                    ]
                
                segments.append(seg_dict)
                full_text.append(segment.text.strip())
            
            result = {
                "text": " ".join(full_text),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "segments": segments
            }
            
            logger.info(f"Transcription complete. Language: {info.language}, "
                       f"Duration: {info.duration:.1f}s, Segments: {len(segments)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": settings.WHISPER_MODEL,
            "device": settings.DEVICE,
            "compute_type": settings.COMPUTE_TYPE,
            "is_loaded": self._model is not None
        }


# Global transcriber instance
transcriber = WhisperTranscriber()
