"""
Transcription history management using SQLite.
Stores transcription results for configurable number of days for re-download.
"""
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(settings.OUTPUT_DIR) / "history.db"

# Settings file for user-configurable options
SETTINGS_PATH = Path(settings.OUTPUT_DIR) / "settings.json"


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class TranscriptionHistory:
    """
    Manages transcription history storage and retrieval.
    Automatically cleans up records older than 90 days.
    """
    
    _instance: Optional['TranscriptionHistory'] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database and create tables if needed"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create transcriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    audio_duration REAL,
                    processing_duration REAL,
                    language TEXT,
                    format TEXT NOT NULL,
                    diarization BOOLEAN DEFAULT FALSE,
                    speakers_count INTEGER,
                    segments_count INTEGER,
                    result_text TEXT NOT NULL,
                    result_json TEXT
                )
            """)
            
            # Create index on created_at for efficient cleanup
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON transcriptions(created_at)
            """)
            
            # Migration: Add processing_duration column if it doesn't exist
            cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'processing_duration' not in columns:
                cursor.execute("ALTER TABLE transcriptions ADD COLUMN processing_duration REAL")
                logger.info("Added processing_duration column to transcriptions table")
            
            conn.commit()
            logger.info(f"History database initialized at {DB_PATH}")
    
    def save_transcription(
        self,
        filename: str,
        file_size: int,
        audio_duration: float,
        language: str,
        format: str,
        diarization: bool,
        speakers_count: int,
        segments_count: int,
        result_text: str,
        result_json: Optional[Dict] = None,
        processing_duration: Optional[float] = None
    ) -> int:
        """
        Save a transcription result to history.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            audio_duration: Audio duration in seconds
            language: Detected/specified language
            format: Output format (text, json, srt, vtt)
            diarization: Whether diarization was enabled
            speakers_count: Number of speakers detected
            segments_count: Number of segments
            result_text: The transcription result (formatted)
            result_json: Full JSON result for re-formatting
            processing_duration: Time taken to process in seconds
            
        Returns:
            ID of the saved transcription
        """
        # Clean up old records first using configured retention days
        retention_days = self.get_retention_days()
        self._cleanup_old_records(days=retention_days)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO transcriptions (
                    filename, file_size, audio_duration, processing_duration, language,
                    format, diarization, speakers_count, segments_count,
                    result_text, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                file_size,
                audio_duration,
                processing_duration,
                language,
                format,
                diarization,
                speakers_count,
                segments_count,
                result_text,
                json.dumps(result_json) if result_json else None
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            
            logger.info(f"Saved transcription to history: {filename} (ID: {record_id})")
            return record_id
    
    def get_transcription(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific transcription by ID.
        
        Args:
            record_id: The transcription ID
            
        Returns:
            Transcription record as dictionary or None
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM transcriptions WHERE id = ?
            """, (record_id,))
            
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def list_transcriptions(
        self, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List transcriptions with pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of transcription records (without full result_text for efficiency)
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id, created_at, filename, file_size, audio_duration,
                    processing_duration, language, format, diarization, 
                    speakers_count, segments_count
                FROM transcriptions 
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_total_count(self) -> int:
        """Get total number of transcriptions in history"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM transcriptions")
            return cursor.fetchone()[0]
    
    def delete_transcription(self, record_id: int) -> bool:
        """
        Delete a specific transcription.
        
        Args:
            record_id: The transcription ID
            
        Returns:
            True if deleted, False if not found
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM transcriptions WHERE id = ?
            """, (record_id,))
            
            conn.commit()
            deleted = cursor.rowcount > 0
            
            if deleted:
                logger.info(f"Deleted transcription from history: ID {record_id}")
            
            return deleted
    
    def update_speaker_names(
        self, 
        record_id: int, 
        speaker_names: Dict[str, str]
    ) -> bool:
        """
        Update speaker names in a transcription record.
        Replaces SPEAKER_XX with the provided names in both result_text and result_json.
        
        Args:
            record_id: The transcription ID
            speaker_names: Dict mapping speaker IDs to names (e.g., {"SPEAKER_00": "Jean"})
            
        Returns:
            True if updated, False if not found
        """
        # Get current record
        record = self.get_transcription(record_id)
        if not record:
            return False
        
        # Update result_text
        result_text = record.get('result_text', '')
        for speaker_id, name in speaker_names.items():
            if name and name.strip():
                result_text = result_text.replace(speaker_id, name.strip())
        
        # Update result_json if present
        result_json = record.get('result_json')
        if result_json and isinstance(result_json, dict):
            # Update segments
            segments = result_json.get('segments', [])
            for segment in segments:
                speaker = segment.get('speaker')
                if speaker and speaker in speaker_names and speaker_names[speaker].strip():
                    segment['speaker'] = speaker_names[speaker].strip()
            
            # Update text field if present
            if 'text' in result_json:
                for speaker_id, name in speaker_names.items():
                    if name and name.strip():
                        result_json['text'] = result_json['text'].replace(speaker_id, name.strip())
        
        # Save updated record
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE transcriptions 
                SET result_text = ?, result_json = ?
                WHERE id = ?
            """, (
                result_text,
                json.dumps(result_json) if result_json else None,
                record_id
            ))
            
            conn.commit()
            updated = cursor.rowcount > 0
            
            if updated:
                logger.info(f"Updated speaker names in history: ID {record_id}")
            
            return updated
    
    def _cleanup_old_records(self, days: int = 90) -> int:
        """
        Remove records older than specified days.
        
        Args:
            days: Number of days to keep records
            
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM transcriptions 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            conn.commit()
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old transcription records")
            
            return deleted_count
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to dictionary"""
        result = dict(row)
        
        # Parse JSON if present
        if 'result_json' in result and result['result_json']:
            try:
                result['result_json'] = json.loads(result['result_json'])
            except json.JSONDecodeError:
                pass
        
        return result
    
    def get_retention_days(self) -> int:
        """
        Get the configured number of days to retain history.
        
        Returns:
            Number of days to keep transcription history
        """
        try:
            if SETTINGS_PATH.exists():
                with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
                    return user_settings.get('retention_days', settings.HISTORY_RETENTION_DAYS)
        except Exception as e:
            logger.warning(f"Failed to read settings: {e}")
        
        return settings.HISTORY_RETENTION_DAYS
    
    def set_retention_days(self, days: int) -> bool:
        """
        Set the number of days to retain history.
        
        Args:
            days: Number of days (minimum 1, maximum 365)
            
        Returns:
            True if successful
        """
        # Validate range
        days = max(1, min(365, days))
        
        try:
            # Load existing settings or create new
            user_settings = {}
            if SETTINGS_PATH.exists():
                with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
            
            user_settings['retention_days'] = days
            
            # Ensure directory exists
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Save settings
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(user_settings, f, indent=2)
            
            logger.info(f"History retention set to {days} days")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False


# Global history instance
history = TranscriptionHistory()
