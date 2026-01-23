"""
Speaker diarization using pyannote.audio
Optimized for NVIDIA RTX GPUs with TF32 and float16 support
Includes chunked processing for long audio files to avoid OOM errors
"""
import logging
import os
import gc
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
    
    def unload(self) -> None:
        """
        Unload the diarization pipeline from GPU to free VRAM.
        Should be called after diarization is complete to allow Whisper
        to use the full GPU memory.
        """
        if self._pipeline is not None:
            logger.info("Unloading diarization pipeline from GPU...")
            
            try:
                import torch
                
                # Delete the pipeline directly
                del self._pipeline
                self._pipeline = None
                
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
                logger.warning(f"Error during pipeline unload: {e}")
                self._pipeline = None
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds"""
        try:
            import torchaudio
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0
    
    def _diarize_chunk(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Diarize a specific chunk of audio.
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            min_speakers: Minimum speakers constraint
            max_speakers: Maximum speakers constraint
            
        Returns:
            List of speaker segments for this chunk
        """
        import torch
        from pyannote.core import Segment
        
        # Build inference parameters
        params = {}
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers
        
        # Run diarization on the specific segment
        with torch.inference_mode():
            # Use pyannote's built-in cropping
            diarization = self._pipeline(
                {"uri": audio_path, "audio": audio_path},
                **params
            ).crop(Segment(start_time, end_time))
        
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker
            })
        
        return timeline
    
    def diarize(
        self, 
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        chunk_duration: float = 300.0,  # 5 minutes chunks
        auto_unload: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file.
        For long audio files, processes in chunks to avoid OOM errors.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional, speeds up if known)
            max_speakers: Maximum number of speakers (optional, speeds up if known)
            chunk_duration: Duration of each chunk in seconds (default 5 minutes)
            auto_unload: Automatically unload model from GPU after diarization
            progress_callback: Optional callback(step, current, total, sub_percent) for progress reporting
                - step: 'loading', 'processing', 'chunk', 'merging'
                - current/total: for chunk-based progress
                - sub_percent: 0.0 to 1.0 for sub-step progress
            
        Returns:
            List of speaker segments:
            [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}, ...]
        """
        # Report loading progress
        if progress_callback:
            progress_callback("loading", 0, 1, 0.0)
        
        self._load_pipeline()
        
        # Report loading complete
        if progress_callback:
            progress_callback("loading", 0, 1, 1.0)
        
        logger.info(f"Starting diarization: {audio_path}")
        if min_speakers or max_speakers:
            logger.info(f"Speaker constraints: min={min_speakers}, max={max_speakers}")
        
        try:
            import torch
            
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            
            # Force re-enable TF32 (pyannote disables it, but it's safe for inference)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Build inference parameters
            params = {}
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers
            
            # For short audio (< 10 minutes), process directly
            # For longer audio, use chunked processing
            if duration > 0 and duration > 600:  # More than 10 minutes
                logger.info(f"Long audio detected ({duration:.1f}s), using chunked processing...")
                timeline = self._diarize_chunked(
                    audio_path, duration, chunk_duration, min_speakers, max_speakers,
                    progress_callback=progress_callback
                )
            else:
                # Standard processing for shorter files
                logger.info("Processing audio in single pass...")
                
                # Report processing start
                if progress_callback:
                    progress_callback("processing", 0, 1, 0.0)
                
                # For short audio, we can't get real progress from pyannote pipeline
                # but we can simulate progress based on estimated duration
                # The pipeline is blocking, so we report progress before and after
                
                # Estimate: pipeline takes roughly 0.2-0.5x realtime on GPU
                estimated_processing_time = duration * 0.3 if duration > 0 else 30
                logger.info(f"Estimated processing time: {estimated_processing_time:.1f}s for {duration:.1f}s audio")
                
                with torch.inference_mode():
                    diarization = self._pipeline(audio_path, **params)
                
                # Report processing complete
                if progress_callback:
                    progress_callback("processing", 1, 1, 1.0)
                
                timeline = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    timeline.append({
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                        "speaker": speaker
                    })
            
            # Report merging step
            if progress_callback:
                progress_callback("merging", 0, 1, 0.5)
            
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
    
    def _diarize_chunked(
        self,
        audio_path: str,
        total_duration: float,
        chunk_duration: float,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process long audio in chunks with overlap to handle speaker continuity.
        
        Uses overlapping windows and merges results to avoid splitting
        speakers at chunk boundaries.
        """
        import torch
        from pyannote.audio import Audio
        from pyannote.core import Segment
        
        # Use 30 second overlap between chunks
        overlap = 30.0
        
        all_segments = []
        chunk_start = 0.0
        chunk_num = 0
        
        # Calculate total number of chunks
        total_chunks = 0
        temp_start = 0.0
        while temp_start < total_duration:
            total_chunks += 1
            temp_start = min(temp_start + chunk_duration, total_duration) - overlap
            if temp_start + overlap >= total_duration:
                break
        
        logger.info(f"Will process {total_chunks} chunks for {total_duration:.1f}s audio")
        
        # Build inference parameters
        params = {}
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers
        
        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_duration, total_duration)
            chunk_num += 1
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks}: {chunk_start:.1f}s - {chunk_end:.1f}s")
            
            # Report chunk progress (using new callback format)
            if progress_callback:
                progress_callback("chunk", chunk_num, total_chunks, 0.0)
            
            try:
                # Clear GPU cache before each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process this chunk
                with torch.inference_mode():
                    # Load and crop audio for this chunk
                    audio = Audio(mono="downmix", sample_rate=16000)
                    waveform, sample_rate = audio.crop(
                        audio_path,
                        Segment(chunk_start, chunk_end)
                    )
                    
                    # Run diarization on the chunk
                    diarization = self._pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate},
                        **params
                    )
                
                # Extract segments and adjust timestamps
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Adjust timestamps to absolute positions
                    abs_start = chunk_start + turn.start
                    abs_end = chunk_start + turn.end
                    
                    # Skip segments in the overlap region (except for last chunk)
                    # This prevents duplicate segments
                    if chunk_start > 0 and abs_start < chunk_start + overlap / 2:
                        continue
                    
                    all_segments.append({
                        "start": round(abs_start, 3),
                        "end": round(abs_end, 3),
                        "speaker": speaker,
                        "chunk_id": chunk_num  # Track which chunk this came from
                    })
                
                # Clean up chunk data
                del waveform
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM on chunk {chunk_num}, trying with smaller chunk...")
                # If OOM, try with smaller chunk (recursive with half size)
                if chunk_duration > 60:
                    sub_segments = self._diarize_chunked(
                        audio_path,
                        chunk_end,
                        chunk_duration / 2,
                        min_speakers,
                        max_speakers
                    )
                    all_segments.extend([s for s in sub_segments if s["start"] >= chunk_start])
                else:
                    raise
            
            # Move to next chunk (with overlap)
            chunk_start = chunk_end - overlap
            if chunk_start + overlap >= total_duration:
                break
        
        # Sort by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Merge speaker labels across chunks (same speaker might have different IDs)
        # This is a simple heuristic - speakers close in time with same ID are merged
        return self._harmonize_speakers(all_segments)
    
    def _harmonize_speakers(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Harmonize speaker labels across chunks.
        Ensures consistent speaker IDs throughout the audio by mapping
        speaker IDs from different chunks to a unified set.
        
        Strategy:
        1. Group segments by chunk
        2. For consecutive chunks, find temporal overlaps to map speakers
        3. Build a global speaker mapping
        4. Apply the mapping to all segments
        """
        if not segments:
            return segments
        
        # Check if we have chunk info (single-pass processing won't have it)
        if not any("chunk_id" in seg for seg in segments):
            return segments
        
        # Group segments by chunk
        chunks: Dict[int, List[Dict[str, Any]]] = {}
        for seg in segments:
            chunk_id = seg.get("chunk_id", 0)
            if chunk_id not in chunks:
                chunks[chunk_id] = []
            chunks[chunk_id].append(seg)
        
        # If only one chunk, no harmonization needed
        if len(chunks) <= 1:
            # Remove chunk_id from output
            return [{k: v for k, v in seg.items() if k != "chunk_id"} for seg in segments]
        
        # Build speaker mapping across chunks
        # global_mapping[chunk_id][local_speaker] = global_speaker
        global_mapping: Dict[int, Dict[str, str]] = {}
        next_global_speaker_id = 0
        global_speakers: Dict[str, int] = {}  # Maps speaker characteristics to global ID
        
        chunk_ids = sorted(chunks.keys())
        
        for i, chunk_id in enumerate(chunk_ids):
            chunk_segs = chunks[chunk_id]
            local_speakers = set(seg["speaker"] for seg in chunk_segs)
            
            if i == 0:
                # First chunk: assign global IDs directly
                global_mapping[chunk_id] = {}
                for speaker in sorted(local_speakers):
                    global_name = f"SPEAKER_{next_global_speaker_id:02d}"
                    global_mapping[chunk_id][speaker] = global_name
                    next_global_speaker_id += 1
            else:
                # Subsequent chunks: map to previous chunk's speakers based on timing
                prev_chunk_id = chunk_ids[i - 1]
                prev_segs = chunks[prev_chunk_id]
                
                # Find the overlap region (last 30s of prev chunk, first 30s of current)
                if prev_segs and chunk_segs:
                    prev_end_time = max(seg["end"] for seg in prev_segs)
                    curr_start_time = min(seg["start"] for seg in chunk_segs)
                    
                    # Get speakers active near the boundary
                    # Look at last 60 seconds of previous chunk
                    prev_boundary_segs = [
                        seg for seg in prev_segs 
                        if seg["end"] > prev_end_time - 60
                    ]
                    # Look at first 60 seconds of current chunk
                    curr_boundary_segs = [
                        seg for seg in chunk_segs 
                        if seg["start"] < curr_start_time + 60
                    ]
                    
                    # Calculate overlap between speakers
                    speaker_overlaps: Dict[str, Dict[str, float]] = {}
                    
                    for curr_seg in curr_boundary_segs:
                        curr_speaker = curr_seg["speaker"]
                        if curr_speaker not in speaker_overlaps:
                            speaker_overlaps[curr_speaker] = {}
                        
                        for prev_seg in prev_boundary_segs:
                            prev_speaker = prev_seg["speaker"]
                            
                            # Calculate temporal proximity (not overlap, but closeness)
                            # Speakers that follow each other closely are likely the same
                            time_gap = abs(curr_seg["start"] - prev_seg["end"])
                            
                            if time_gap < 30:  # Within 30 seconds
                                # Weight by inverse of time gap
                                weight = 1.0 / (1.0 + time_gap)
                                prev_global = global_mapping[prev_chunk_id].get(prev_speaker)
                                if prev_global:
                                    if prev_global not in speaker_overlaps[curr_speaker]:
                                        speaker_overlaps[curr_speaker][prev_global] = 0
                                    speaker_overlaps[curr_speaker][prev_global] += weight
                    
                    # Map current speakers to global speakers
                    global_mapping[chunk_id] = {}
                    used_global_speakers = set()
                    
                    # Sort current speakers by their total overlap score (most confident first)
                    sorted_speakers = sorted(
                        local_speakers,
                        key=lambda s: max(speaker_overlaps.get(s, {}).values(), default=0),
                        reverse=True
                    )
                    
                    for speaker in sorted_speakers:
                        overlaps = speaker_overlaps.get(speaker, {})
                        
                        # Find best matching global speaker not yet used
                        best_match = None
                        best_score = 0
                        
                        for global_speaker, score in sorted(overlaps.items(), key=lambda x: -x[1]):
                            if global_speaker not in used_global_speakers and score > best_score:
                                best_match = global_speaker
                                best_score = score
                        
                        if best_match and best_score > 0.1:
                            # Use existing global speaker
                            global_mapping[chunk_id][speaker] = best_match
                            used_global_speakers.add(best_match)
                        else:
                            # New speaker, assign new global ID
                            global_name = f"SPEAKER_{next_global_speaker_id:02d}"
                            global_mapping[chunk_id][speaker] = global_name
                            next_global_speaker_id += 1
                            used_global_speakers.add(global_name)
        
        # Apply mapping to all segments
        harmonized = []
        for seg in segments:
            chunk_id = seg.get("chunk_id", 0)
            local_speaker = seg["speaker"]
            
            # Get global speaker name
            if chunk_id in global_mapping and local_speaker in global_mapping[chunk_id]:
                global_speaker = global_mapping[chunk_id][local_speaker]
            else:
                global_speaker = local_speaker
            
            # Create new segment without chunk_id
            new_seg = {k: v for k, v in seg.items() if k != "chunk_id"}
            new_seg["speaker"] = global_speaker
            harmonized.append(new_seg)
        
        # Log mapping info
        unique_speakers = set(seg["speaker"] for seg in harmonized)
        logger.info(f"Harmonized speakers across {len(chunks)} chunks: {len(unique_speakers)} unique speakers")
        
        return harmonized
    
    def merge_with_transcription(
        self,
        transcription_segments: List[Dict[str, Any]],
        diarization_timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge transcription segments with speaker diarization.
        
        Assigns the dominant speaker to each transcription segment based on
        overlap with diarization timeline. Falls back to nearest speaker
        if no direct overlap is found (can happen with chunked processing).
        
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
        max_distance: float = 5.0  # Max 5 seconds gap
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
            "hf_token_configured": bool(settings.HF_TOKEN),
            "pipeline_loaded": self._pipeline is not None
        }


# Global diarizer instance
diarizer = SpeakerDiarizer()
