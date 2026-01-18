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
