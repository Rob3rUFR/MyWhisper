"""
Utility functions for file validation, formatting, and conversions
"""
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"}


def validate_audio_file(filename: str, file_size: int, max_size: int) -> Tuple[bool, Optional[str]]:
    """
    Validate an audio file by extension and size.
    
    Args:
        filename: Name of the file
        file_size: Size of the file in bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Format non supporté. Formats acceptés: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    
    # Check size
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"Fichier trop volumineux. Taille max: {max_mb:.0f}MB"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename with hash prefix
    """
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Generate safe name with hash
    name, ext = os.path.splitext(filename)
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    
    # Keep only alphanumeric chars and some safe chars
    safe_name = "".join(c for c in name[:50] if c.isalnum() or c in "._- ")
    safe_name = f"{hash_suffix}_{safe_name}{ext.lower()}"
    
    return safe_name


def format_timestamp_srt(seconds: float) -> str:
    """
    Format timestamp in SRT format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Format timestamp in VTT format (HH:MM:SS.mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    return format_timestamp_srt(seconds).replace(',', '.')


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert transcription segments to SRT format.
    
    Args:
        segments: List of segment dictionaries with start, end, text
        
    Returns:
        SRT formatted string
    """
    srt_lines = []
    
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg['start'])
        end = format_timestamp_srt(seg['end'])
        text = seg['text'].strip()
        
        # Add speaker label if present
        if 'speaker' in seg and seg['speaker']:
            text = f"[{seg['speaker']}] {text}"
        
        srt_lines.append(str(i))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line
    
    return "\n".join(srt_lines)


def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert transcription segments to VTT format.
    
    Args:
        segments: List of segment dictionaries with start, end, text
        
    Returns:
        VTT formatted string
    """
    vtt_lines = ["WEBVTT", ""]
    
    for seg in segments:
        start = format_timestamp_vtt(seg['start'])
        end = format_timestamp_vtt(seg['end'])
        text = seg['text'].strip()
        
        # Add speaker label if present
        if 'speaker' in seg and seg['speaker']:
            text = f"[{seg['speaker']}] {text}"
        
        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(text)
        vtt_lines.append("")
    
    return "\n".join(vtt_lines)


def segments_to_text(segments: List[Dict[str, Any]], include_speakers: bool = False) -> str:
    """
    Convert transcription segments to plain text.
    
    Args:
        segments: List of segment dictionaries
        include_speakers: Whether to include speaker labels
        
    Returns:
        Plain text transcription
    """
    lines = []
    current_speaker = None
    
    for seg in segments:
        text = seg['text'].strip()
        speaker = seg.get('speaker')
        
        if include_speakers and speaker and speaker != current_speaker:
            lines.append(f"\n[{speaker}]")
            current_speaker = speaker
        
        lines.append(text)
    
    return " ".join(lines).strip()


def get_file_extension(filename: str) -> str:
    """
    Get lowercase file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        Lowercase extension without dot
    """
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''


# Onomatopées et mots courts à éviter pour les extraits audio
FILLER_WORDS = {
    # French
    "euh", "hein", "hmm", "ah", "oh", "eh", "bah", "ben", "hum", "ouais", 
    "oui", "non", "ok", "okay", "d'accord", "voilà", "bon", "bref",
    "alors", "donc", "quoi", "enfin", "hé", "ho", "pff", "mouais",
    # English
    "uh", "um", "yeah", "yes", "no", "ok", "okay", "well", "so", "like",
    "right", "alright", "hmm", "ah", "oh", "hey", "hi", "bye",
    # Spanish
    "eh", "pues", "bueno", "sí", "no", "vale", "ajá",
    # German
    "ähm", "äh", "ja", "nein", "also", "gut", "na",
}


def _count_meaningful_words(text: str) -> int:
    """Count meaningful words in text (excluding fillers)."""
    if not text:
        return 0
    words = [w.strip(".,!?;:'\"()[]«»…-–—") for w in text.lower().split()]
    return len([w for w in words if w and w not in FILLER_WORDS and len(w) > 1])


def _calculate_speaker_confidence(
    seg_start: float, 
    seg_end: float, 
    speaker: str,
    diarization_timeline: List[Dict[str, Any]]
) -> float:
    """
    Calculate confidence that a segment truly belongs to the given speaker.
    
    Returns a score from 0 to 1:
    - 1.0 = 100% of the segment is covered by this speaker in diarization
    - 0.0 = No overlap with this speaker
    
    Args:
        seg_start: Segment start time
        seg_end: Segment end time
        speaker: Speaker ID to check
        diarization_timeline: Original diarization segments
        
    Returns:
        Confidence score (0 to 1)
    """
    if not diarization_timeline:
        return 0.5  # No diarization data, neutral confidence
    
    seg_duration = seg_end - seg_start
    if seg_duration <= 0:
        return 0.0
    
    speaker_overlap = 0.0
    other_overlap = 0.0
    
    for dia_seg in diarization_timeline:
        dia_start = dia_seg["start"]
        dia_end = dia_seg["end"]
        dia_speaker = dia_seg["speaker"]
        
        # Calculate overlap with this diarization segment
        overlap_start = max(seg_start, dia_start)
        overlap_end = min(seg_end, dia_end)
        
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            
            if dia_speaker == speaker:
                speaker_overlap += overlap_duration
            else:
                other_overlap += overlap_duration
    
    # Confidence = speaker overlap / segment duration
    # Penalize if other speakers also overlap
    total_overlap = speaker_overlap + other_overlap
    
    if total_overlap == 0:
        return 0.0
    
    # Pure speaker score (how much of overlapping audio is this speaker)
    purity = speaker_overlap / total_overlap if total_overlap > 0 else 0
    
    # Coverage score (how much of the segment has any speaker)
    coverage = min(total_overlap / seg_duration, 1.0)
    
    # Combined confidence: both purity and coverage matter
    return purity * coverage


def _merge_consecutive_segments(
    segments: List[Dict[str, Any]], 
    speaker: str,
    max_gap: float = 0.5,
    target_duration: float = 5.0,
    max_duration: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments from the same speaker into longer chunks.
    
    This creates more robust samples by combining multiple short segments
    into longer, more representative samples.
    
    Args:
        segments: All transcription segments (will filter by speaker)
        speaker: Speaker to merge segments for
        max_gap: Maximum gap between segments to merge (seconds)
        target_duration: Target duration for merged segments
        max_duration: Maximum duration for merged segments
        
    Returns:
        List of merged segments for this speaker
    """
    # Filter and sort segments for this speaker
    speaker_segs = sorted(
        [s for s in segments if s.get("speaker") == speaker],
        key=lambda x: x["start"]
    )
    
    if not speaker_segs:
        return []
    
    merged = []
    current = None
    
    for seg in speaker_segs:
        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": speaker,
                "segment_count": 1
            }
        else:
            gap = seg["start"] - current["end"]
            current_duration = current["end"] - current["start"]
            new_duration = seg["end"] - current["start"]
            
            # Merge if: gap is small AND we won't exceed max duration
            if gap <= max_gap and new_duration <= max_duration:
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"].strip()
                current["segment_count"] += 1
            else:
                # Save current and start new
                merged.append(current)
                current = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "speaker": speaker,
                    "segment_count": 1
                }
    
    # Don't forget the last segment
    if current:
        merged.append(current)
    
    return merged


def _time_ranges_overlap(start1: float, end1: float, start2: float, end2: float, min_overlap: float = 0.5) -> bool:
    """Check if two time ranges overlap significantly."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return False
    
    overlap_duration = overlap_end - overlap_start
    min_duration = min(end1 - start1, end2 - start2)
    
    # Consider overlapping if more than 50% of smaller segment overlaps
    return overlap_duration > (min_duration * min_overlap)


def select_speaker_samples(
    segments: List[Dict[str, Any]],
    diarization_timeline: List[Dict[str, Any]] = None,
    sample_duration: float = 5.0,
    min_duration: float = 2.5,
    max_duration: float = 10.0
) -> Dict[str, Dict[str, Any]]:
    """
    Select the best audio sample for each speaker from transcription segments.
    
    Uses a multi-criteria scoring system:
    1. Speaker confidence: How certain we are this is the right speaker
    2. Content quality: Meaningful words vs fillers
    3. Duration fitness: Close to target duration
    4. Segment purity: Prefers merged consecutive segments
    
    Ensures no two speakers have overlapping audio samples.
    
    Args:
        segments: List of transcription segments with speaker labels
        diarization_timeline: Original diarization segments for confidence scoring
        sample_duration: Target duration for samples
        min_duration: Minimum acceptable duration
        max_duration: Maximum duration to consider
        
    Returns:
        Dict mapping speaker ID to best sample info
    """
    if not segments:
        return {}
    
    # Get unique speakers
    speakers = set(seg.get("speaker") for seg in segments if seg.get("speaker"))
    
    samples = {}
    used_time_ranges = []  # Track already selected time ranges to avoid duplicates
    
    # Sort speakers to process consistently
    speakers = sorted(speakers)
    
    for speaker in speakers:
        # Create merged segments for this speaker
        merged_segs = _merge_consecutive_segments(
            segments, speaker, 
            max_gap=0.5, 
            target_duration=sample_duration,
            max_duration=max_duration
        )
        
        # Also consider original segments
        original_segs = [
            {**s, "segment_count": 1} 
            for s in segments 
            if s.get("speaker") == speaker
        ]
        
        # Combine and score all candidates
        all_candidates = merged_segs + original_segs
        
        best_sample = None
        best_score = -1
        
        for seg in all_candidates:
            duration = seg["end"] - seg["start"]
            
            # Skip if duration out of range
            if duration < min_duration or duration > max_duration:
                continue
            
            text = seg.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            
            # Skip if this time range overlaps with an already selected sample
            is_duplicate = False
            for used_start, used_end in used_time_ranges:
                if _time_ranges_overlap(seg["start"], seg["end"], used_start, used_end):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # === SCORING ===
            
            # 1. Speaker confidence (0-1) - weight: 35%
            if diarization_timeline:
                confidence = _calculate_speaker_confidence(
                    seg["start"], seg["end"], speaker, diarization_timeline
                )
            else:
                confidence = 0.7  # Default if no diarization data
            
            # Skip low confidence segments
            if confidence < 0.6:
                continue
            
            # 2. Content quality (0-1) - weight: 30%
            meaningful_words = _count_meaningful_words(text)
            word_score = min(meaningful_words / 8, 1.0)  # Cap at 8 words
            
            # Skip segments with too few meaningful words
            if meaningful_words < 3:
                continue
            
            # 3. Duration fitness (0-1) - weight: 20%
            # Prefer segments close to target duration
            duration_diff = abs(duration - sample_duration)
            duration_score = max(0, 1.0 - (duration_diff / sample_duration))
            
            # 4. Segment quality bonus (0-0.15) - weight: 15%
            # Prefer merged segments (more stable) but not too many
            segment_count = seg.get("segment_count", 1)
            if segment_count >= 2 and segment_count <= 4:
                merge_bonus = 0.15  # Good merge
            elif segment_count == 1:
                merge_bonus = 0.10  # Single segment, okay
            else:
                merge_bonus = 0.05  # Too many merged, might be fragmented
            
            # Calculate final score
            score = (
                confidence * 0.35 +
                word_score * 0.30 +
                duration_score * 0.20 +
                merge_bonus
            )
            
            if score > best_score:
                best_score = score
                best_sample = {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "text": text[:200] + ("..." if len(text) > 200 else ""),
                    "duration": round(duration, 2),
                    "confidence": round(confidence, 2),
                    "score": round(score, 3)
                }
        
        # Fallback: if no good sample found, take the longest segment
        if best_sample is None:
            speaker_segs = sorted(
                [s for s in segments if s.get("speaker") == speaker],
                key=lambda x: x["end"] - x["start"],
                reverse=True
            )
            
            for seg in speaker_segs:
                duration = seg["end"] - seg["start"]
                text = seg.get("text", "").strip()
                
                # Check for duplicates even in fallback
                is_duplicate = False
                for used_start, used_end in used_time_ranges:
                    if _time_ranges_overlap(seg["start"], seg["end"], used_start, used_end):
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                if duration >= 1.5 and len(text) > 5:
                    best_sample = {
                        "start": round(seg["start"], 3),
                        "end": round(seg["end"], 3),
                        "text": text[:200],
                        "duration": round(duration, 2),
                        "confidence": 0.5,
                        "score": 0.0
                    }
                    logger.warning(f"Using fallback sample for {speaker}")
                    break
        
        if best_sample:
            # Register this time range as used to prevent duplicates
            used_time_ranges.append((best_sample["start"], best_sample["end"]))
            
            samples[speaker] = best_sample
            logger.info(
                f"Selected sample for {speaker}: "
                f"{best_sample['duration']:.1f}s, "
                f"confidence={best_sample['confidence']:.0%}, "
                f"score={best_sample['score']:.2f}"
            )
    
    return samples