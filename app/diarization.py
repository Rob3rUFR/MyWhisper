"""
Speaker diarization using pyannote.audio
Optimized for NVIDIA RTX GPUs with TF32 and float16 support
"""
import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy import to avoid loading pyannote if not needed
Pipeline = None


def _load_pyannote():
    """Lazy load pyannote to reduce startup time if not needed"""
    global Pipeline
    if Pipeline is None:
        # Patches are already applied via app.patches imported in main.py
        from pyannote.audio import Pipeline as PyAnnotePipeline
        Pipeline = PyAnnotePipeline


def _optimize_torch_for_inference():
    """Apply PyTorch optimizations for faster inference on modern GPUs"""
    import torch
    
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix operations on Ampere+ GPUs (RTX 30xx, 40xx, 50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        
        logger.info("Applied PyTorch optimizations: TF32=True, cudnn.benchmark=True")


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote/speaker-diarization-3.1
    Requires Hugging Face token for model access.
    """
    
    _instance: Optional['SpeakerDiarizer'] = None
    _pipeline = None
    _available: Optional[bool] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize diarizer (pipeline loaded on first use)"""
        pass
    
    @property
    def is_available(self) -> bool:
        """Check if diarization is available (HF token configured)"""
        if self._available is None:
            self._available = bool(settings.HF_TOKEN) and settings.ENABLE_DIARIZATION
            if not self._available:
                if not settings.HF_TOKEN:
                    logger.warning("Diarization unavailable: HF_TOKEN not configured")
                elif not settings.ENABLE_DIARIZATION:
                    logger.info("Diarization disabled by configuration")
        return self._available
    
    def _load_pipeline(self) -> None:
        """Load the diarization pipeline with GPU optimizations"""
        if self._pipeline is not None:
            return
            
        if not self.is_available:
            raise RuntimeError("Diarization not available. Check HF_TOKEN configuration.")
        
        logger.info("Loading diarization pipeline...")
        _load_pyannote()
        
        cache_dir = Path(settings.MODEL_DIR) / "pyannote"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.HF_TOKEN,
                cache_dir=str(cache_dir)
            )
            
            # Move to GPU if available
            if settings.DEVICE == "cuda":
                import torch
                if torch.cuda.is_available():
                    # Apply optimizations before moving to GPU
                    _optimize_torch_for_inference()
                    
                    device = torch.device("cuda")
                    self._pipeline.to(device)
                    
                    # Note: We keep float32 for the models to ensure maximum quality
                    # TF32 is enabled which provides ~3x speedup without quality loss
                    logger.info("Diarization pipeline loaded on GPU (TF32 enabled)")
                else:
                    logger.warning("CUDA requested but not available, using CPU")
            else:
                logger.info("Diarization pipeline loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            self._available = False
            raise
    
    def diarize(
        self, 
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional, speeds up if known)
            max_speakers: Maximum number of speakers (optional, speeds up if known)
            
        Returns:
            List of speaker segments:
            [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
        """
        self._load_pipeline()
        
        logger.info(f"Starting diarization: {audio_path}")
        if min_speakers or max_speakers:
            logger.info(f"Speaker constraints: min={min_speakers}, max={max_speakers}")
        
        try:
            import torch
            
            # Build inference parameters
            params = {}
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers
            
            # Force re-enable TF32 (pyannote disables it, but it's safe for inference)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Run with optimized inference
            with torch.inference_mode():
                diarization = self._pipeline(audio_path, **params)
            
            timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                timeline.append({
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker
                })
            
            # Count unique speakers
            speakers = set(seg["speaker"] for seg in timeline)
            logger.info(f"Diarization complete. Found {len(speakers)} speakers, "
                       f"{len(timeline)} segments")
            
            return timeline
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
    
    def merge_with_transcription(
        self,
        transcription_segments: List[Dict[str, Any]],
        diarization_timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge transcription segments with speaker diarization.
        
        Assigns the dominant speaker to each transcription segment based on
        overlap with diarization timeline.
        
        Args:
            transcription_segments: Segments from Whisper transcription
            diarization_timeline: Segments from speaker diarization
            
        Returns:
            Transcription segments with speaker labels added
        """
        logger.info("Merging transcription with diarization...")
        
        merged_segments = []
        
        for seg in transcription_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_copy = dict(seg)
            
            # Find overlapping diarization segments
            overlaps = {}
            
            for dia_seg in diarization_timeline:
                dia_start = dia_seg["start"]
                dia_end = dia_seg["end"]
                speaker = dia_seg["speaker"]
                
                # Calculate overlap
                overlap_start = max(seg_start, dia_start)
                overlap_end = min(seg_end, dia_end)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlaps[speaker] = overlaps.get(speaker, 0) + overlap_duration
            
            # Assign dominant speaker
            if overlaps:
                dominant_speaker = max(overlaps, key=overlaps.get)
                seg_copy["speaker"] = dominant_speaker
            else:
                seg_copy["speaker"] = "UNKNOWN"
            
            merged_segments.append(seg_copy)
        
        logger.info(f"Merged {len(merged_segments)} segments with speaker labels")
        return merged_segments
    
    def get_info(self) -> Dict[str, Any]:
        """Get diarization service info"""
        return {
            "available": self.is_available,
            "enabled": settings.ENABLE_DIARIZATION,
            "hf_token_configured": bool(settings.HF_TOKEN),
            "pipeline_loaded": self._pipeline is not None
        }


# Global diarizer instance
diarizer = SpeakerDiarizer()
