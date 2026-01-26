"""
Speaker diarization using NVIDIA NeMo Sortformer.
Sortformer is an end-to-end Transformer-based model that handles long audio natively.
"""
import logging
import os
import gc
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


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
    Speaker diarization using NVIDIA NeMo Sortformer.
    
    Sortformer is an end-to-end speaker diarization model that:
    - Processes long audio files natively without manual chunking
    - Produces speaker labels sorted by arrival time
    - Handles speaker counting automatically
    """
    
    _instance: Optional['SpeakerDiarizer'] = None
    _model = None
    _available: Optional[bool] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize diarizer (model loaded on first use)"""
        pass
    
    @property
    def is_available(self) -> bool:
        """Check if diarization is available"""
        if self._available is None:
            self._available = settings.ENABLE_DIARIZATION
            if not self._available:
                logger.info("Diarization disabled by configuration")
        return self._available
    
    def _load_model(self) -> None:
        """Load the NeMo Sortformer model"""
        if self._model is not None:
            return
            
        if not self.is_available:
            raise RuntimeError("Diarization not available. Check ENABLE_DIARIZATION configuration.")
        
        logger.info("Loading NeMo Sortformer diarization model...")
        
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
            import torch
            
            # Apply optimizations before loading
            _optimize_torch_for_inference()
            
            # Create cache directory
            cache_dir = Path(settings.MODEL_DIR) / "nemo"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load pre-trained Sortformer model from NGC
            # Model supports up to 4 speakers by default
            self._model = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_sortformer_4spk-v1"
            )
            
            # Move to GPU if available
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("Sortformer model loaded on GPU")
            else:
                logger.info("Sortformer model loaded on CPU")
            
            # Set to evaluation mode
            self._model.eval()
                
        except Exception as e:
            logger.error(f"Failed to load Sortformer model: {e}")
            self._available = False
            raise
    
    def unload(self) -> None:
        """
        Unload the diarization model from GPU to free VRAM.
        Should be called after diarization is complete to allow Whisper
        to use the full GPU memory.
        """
        if self._model is not None:
            logger.info("Unloading Sortformer model from GPU...")
            
            try:
                import torch
                
                # Delete the model
                del self._model
                self._model = None
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache to actually free the VRAM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Log memory status
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    cached = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU memory after unload: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
                    
            except Exception as e:
                logger.warning(f"Error during model unload: {e}")
                self._model = None
    
    def diarize(
        self, 
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        chunk_duration: float = 300.0,  # Kept for API compatibility, not used by Sortformer
        auto_unload: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file using Sortformer.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional hint)
            max_speakers: Maximum number of speakers (optional, max 4 for default model)
            chunk_duration: Ignored - Sortformer handles long audio natively
            auto_unload: Automatically unload model from GPU after diarization
            progress_callback: Optional callback(step, current, total, sub_percent) for progress
            
        Returns:
            List of speaker segments:
            [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
        """
        import torch
        
        # Report loading progress
        if progress_callback:
            progress_callback("loading", 0, 1, 0.0)
        
        self._load_model()
        
        # Report loading complete
        if progress_callback:
            progress_callback("loading", 0, 1, 1.0)
        
        logger.info(f"Starting Sortformer diarization: {audio_path}")
        if min_speakers or max_speakers:
            logger.info(f"Speaker hints: min={min_speakers}, max={max_speakers}")
        
        try:
            # Report processing start
            if progress_callback:
                progress_callback("processing", 0, 1, 0.0)
            
            # Run diarization with Sortformer
            with torch.inference_mode():
                # Sortformer diarize method returns RTTM-style annotations
                # Parameters for speaker count hints
                params = {}
                if max_speakers is not None and max_speakers > 0:
                    params['num_speakers'] = min(max_speakers, 4)  # Model supports max 4
                
                # Run inference
                annotations = self._model.diarize(
                    audio=[audio_path],
                    batch_size=1,
                    **params
                )
            
            # Report processing complete
            if progress_callback:
                progress_callback("processing", 1, 1, 1.0)
            
            # Convert annotations to timeline format
            if progress_callback:
                progress_callback("merging", 0, 1, 0.5)
            
            timeline = self._convert_annotations_to_timeline(annotations, audio_path)
            
            # Count unique speakers
            speakers = set(seg["speaker"] for seg in timeline)
            logger.info(f"Diarization complete. Found {len(speakers)} speakers, "
                       f"{len(timeline)} segments")
            
            # Report merging complete
            if progress_callback:
                progress_callback("merging", 0, 1, 1.0)
            
            return timeline
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
        finally:
            # Unload model from GPU to free memory for other operations
            if auto_unload:
                self.unload()
    
    def _convert_annotations_to_timeline(
        self, 
        annotations: Any,
        audio_path: str
    ) -> List[Dict[str, Any]]:
        """
        Convert NeMo Sortformer annotations to timeline format.
        
        Sortformer returns annotations in RTTM format or as annotation objects.
        We convert these to our standard timeline format.
        """
        timeline = []
        
        try:
            # Handle different output formats from Sortformer
            if isinstance(annotations, dict):
                # Dict format: {audio_path: annotation}
                annotation = annotations.get(audio_path, annotations.get(list(annotations.keys())[0]))
            elif isinstance(annotations, list) and len(annotations) > 0:
                annotation = annotations[0]
            else:
                annotation = annotations
            
            # If annotation is a pyannote Annotation object (NeMo can return this)
            if hasattr(annotation, 'itertracks'):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    timeline.append({
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                        "speaker": self._normalize_speaker_label(speaker)
                    })
            # If it's a list of segments
            elif isinstance(annotation, list):
                for seg in annotation:
                    if isinstance(seg, dict):
                        timeline.append({
                            "start": round(seg.get("start", seg.get("onset", 0)), 3),
                            "end": round(seg.get("end", seg.get("offset", 0)), 3),
                            "speaker": self._normalize_speaker_label(
                                seg.get("speaker", seg.get("label", "SPEAKER_00"))
                            )
                        })
                    elif hasattr(seg, 'start') and hasattr(seg, 'end'):
                        timeline.append({
                            "start": round(seg.start, 3),
                            "end": round(seg.end, 3),
                            "speaker": self._normalize_speaker_label(
                                getattr(seg, 'speaker', getattr(seg, 'label', 'SPEAKER_00'))
                            )
                        })
            # If it's RTTM string format
            elif isinstance(annotation, str):
                timeline = self._parse_rttm(annotation)
            
        except Exception as e:
            logger.warning(f"Error converting annotations: {e}")
        
        # Sort by start time
        timeline.sort(key=lambda x: x["start"])
        
        # Renumber speakers to be consecutive
        timeline = self._renumber_speakers(timeline)
        
        return timeline
    
    def _parse_rttm(self, rttm_content: str) -> List[Dict[str, Any]]:
        """Parse RTTM format string to timeline."""
        timeline = []
        
        for line in rttm_content.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                timeline.append({
                    "start": round(start, 3),
                    "end": round(start + duration, 3),
                    "speaker": self._normalize_speaker_label(speaker)
                })
        
        return timeline
    
    def _normalize_speaker_label(self, speaker: str) -> str:
        """Normalize speaker label to SPEAKER_XX format."""
        if speaker.startswith("SPEAKER_"):
            return speaker
        
        # Extract number if present
        import re
        match = re.search(r'(\d+)', str(speaker))
        if match:
            num = int(match.group(1))
            return f"SPEAKER_{num:02d}"
        
        # Default
        return "SPEAKER_00"
    
    def _renumber_speakers(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Renumber speakers to be consecutive (SPEAKER_00, SPEAKER_01, ...).
        Preserves order of first appearance.
        """
        if not segments:
            return segments
        
        # Find order of first appearance
        speaker_first_time = {}
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in speaker_first_time:
                speaker_first_time[speaker] = seg["start"]
        
        # Sort by first appearance time
        speakers_ordered = sorted(speaker_first_time.keys(), key=lambda s: speaker_first_time[s])
        
        # Create renumbering map
        renumber_map = {}
        for i, speaker in enumerate(speakers_ordered):
            new_name = f"SPEAKER_{i:02d}"
            if speaker != new_name:
                renumber_map[speaker] = new_name
        
        if not renumber_map:
            return segments
        
        # Apply renumbering
        result = []
        for seg in segments:
            new_seg = dict(seg)
            if seg["speaker"] in renumber_map:
                new_seg["speaker"] = renumber_map[seg["speaker"]]
            result.append(new_seg)
        
        return result
    
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
        last_speaker = None
        unknown_count = 0
        
        for seg in transcription_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_mid = (seg_start + seg_end) / 2
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
                last_speaker = dominant_speaker
            else:
                # No direct overlap found - try to find nearest diarization segment
                nearest_speaker = self._find_nearest_speaker(seg_mid, diarization_timeline)
                
                if nearest_speaker:
                    seg_copy["speaker"] = nearest_speaker
                    last_speaker = nearest_speaker
                elif last_speaker:
                    # Use the last known speaker as fallback
                    seg_copy["speaker"] = last_speaker
                else:
                    # Absolute fallback - should rarely happen
                    seg_copy["speaker"] = "SPEAKER_00"
                    unknown_count += 1
            
            merged_segments.append(seg_copy)
        
        if unknown_count > 0:
            logger.warning(f"Could not determine speaker for {unknown_count} segments (used SPEAKER_00)")
        
        logger.info(f"Merged {len(merged_segments)} segments with speaker labels")
        return merged_segments
    
    def _find_nearest_speaker(
        self, 
        timestamp: float, 
        diarization_timeline: List[Dict[str, Any]],
        max_distance: float = 5.0
    ) -> Optional[str]:
        """
        Find the nearest speaker to a given timestamp.
        
        Args:
            timestamp: The timestamp to find a speaker for
            diarization_timeline: List of diarization segments
            max_distance: Maximum allowed time distance in seconds
            
        Returns:
            Speaker ID or None if no speaker found within max_distance
        """
        if not diarization_timeline:
            return None
        
        nearest_speaker = None
        min_distance = float('inf')
        
        for dia_seg in diarization_timeline:
            dia_start = dia_seg["start"]
            dia_end = dia_seg["end"]
            speaker = dia_seg["speaker"]
            
            # Check if timestamp is within the segment
            if dia_start <= timestamp <= dia_end:
                return speaker
            
            # Calculate distance to segment
            if timestamp < dia_start:
                distance = dia_start - timestamp
            else:
                distance = timestamp - dia_end
            
            if distance < min_distance:
                min_distance = distance
                nearest_speaker = speaker
        
        # Only return if within max_distance
        if min_distance <= max_distance:
            return nearest_speaker
        
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get diarization service info"""
        return {
            "available": self.is_available,
            "enabled": settings.ENABLE_DIARIZATION,
            "model": "nvidia/diar_sortformer_4spk-v1",
            "model_loaded": self._model is not None
        }


# Global diarizer instance
diarizer = SpeakerDiarizer()
