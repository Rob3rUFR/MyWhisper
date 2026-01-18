"""
Compatibility patches for PyTorch 2.6+ and other dependencies.
This module MUST be imported FIRST before any other imports.
"""
import logging
import sys

logger = logging.getLogger(__name__)


def apply_all_patches():
    """Apply all necessary compatibility patches."""
    _patch_torch_load()
    _patch_torchaudio()
    _patch_pyannote_tf32()


def _patch_torch_load():
    """
    Patch torch.load for PyTorch 2.6+ compatibility.
    
    PyTorch 2.6 changed weights_only default to True, breaking
    older model checkpoints (pyannote, speechbrain, etc.).
    
    This patch is aggressive - it patches both torch.load and
    torch.serialization._legacy_load to ensure all code paths use
    weights_only=False.
    """
    try:
        import torch
        import torch.serialization
        
        # Store originals
        _original_load = torch.load
        
        def _patched_load(*args, **kwargs):
            # Force weights_only=False
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        
        # Patch at module level
        torch.load = _patched_load
        
        # Also patch in serialization module directly
        if hasattr(torch.serialization, 'load'):
            torch.serialization.load = _patched_load
        
        # Patch in sys.modules to catch any cached references
        if 'torch' in sys.modules:
            sys.modules['torch'].load = _patched_load
        
        print("[PATCH] Applied torch.load patch (weights_only=False)", flush=True)
        
    except Exception as e:
        print(f"[PATCH] Warning: Could not patch torch.load: {e}", flush=True)


def _patch_torchaudio():
    """
    Patch torchaudio for compatibility with pyannote.audio.
    
    torchaudio >= 2.5 removed several APIs that pyannote still uses.
    """
    try:
        import torchaudio
        
        # Patch AudioMetaData
        if not hasattr(torchaudio, 'AudioMetaData'):
            try:
                from torchaudio._backend.utils import AudioMetaData
                torchaudio.AudioMetaData = AudioMetaData
            except ImportError:
                from dataclasses import dataclass
                
                @dataclass
                class AudioMetaData:
                    sample_rate: int
                    num_frames: int
                    num_channels: int
                    bits_per_sample: int = 16
                    encoding: str = "PCM_S"
                
                torchaudio.AudioMetaData = AudioMetaData
        
        # Patch torchaudio.info if missing
        if not hasattr(torchaudio, 'info'):
            import subprocess
            import json
            import soundfile as sf
            
            def _patched_info(filepath, backend=None, format=None):
                """Get audio file info using ffprobe (supports video files) or soundfile."""
                filepath = str(filepath)
                
                class AudioInfo:
                    def __init__(self, sample_rate, num_frames, num_channels, duration, bits_per_sample=16, encoding="PCM_S"):
                        self.sample_rate = sample_rate
                        self.num_frames = num_frames
                        self.num_channels = num_channels
                        self.bits_per_sample = bits_per_sample
                        self.encoding = encoding
                        self.duration = duration
                
                # Try ffprobe first (supports video files like MP4)
                try:
                    cmd = [
                        'ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_streams', '-select_streams', 'a:0', filepath
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        data = json.loads(result.stdout)
                        if data.get('streams'):
                            stream = data['streams'][0]
                            sample_rate = int(stream.get('sample_rate', 16000))
                            channels = int(stream.get('channels', 1))
                            duration = float(stream.get('duration', 0))
                            num_frames = int(duration * sample_rate)
                            return AudioInfo(sample_rate, num_frames, channels, duration)
                except Exception:
                    pass
                
                # Fallback to soundfile for pure audio files
                try:
                    info = sf.info(filepath)
                    return AudioInfo(
                        info.samplerate, info.frames, info.channels, 
                        info.duration, 16, info.subtype
                    )
                except Exception as e:
                    raise RuntimeError(f"Cannot get audio info for {filepath}: {e}")
            
            torchaudio.info = _patched_info
            print("[PATCH] Added torchaudio.info using ffprobe/soundfile", flush=True)
        
        # Patch audio backend functions
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ["soundfile", "sox"]
        
        if not hasattr(torchaudio, 'get_audio_backend'):
            torchaudio.get_audio_backend = lambda: "soundfile"
        
        if not hasattr(torchaudio, 'set_audio_backend'):
            torchaudio.set_audio_backend = lambda x: None
        
        print("[PATCH] Applied torchaudio compatibility patches", flush=True)
        
    except Exception as e:
        print(f"[PATCH] Warning: Could not patch torchaudio: {e}", flush=True)


def _patch_pyannote_tf32():
    """
    Patch pyannote to keep TF32 enabled for faster inference.
    
    pyannote.audio disables TF32 by default for reproducibility,
    but TF32 is safe for inference and provides ~2-3x speedup.
    """
    try:
        import warnings
        import types
        import sys
        
        # Suppress the pyannote TF32 warning
        warnings.filterwarnings(
            'ignore', 
            message='.*TensorFloat-32.*',
            category=UserWarning
        )
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module='pyannote.*'
        )
        
        # Create a fake reproducibility module with all expected functions
        fake_module = types.ModuleType('pyannote.audio.utils.reproducibility')
        
        # Define ReproducibilityWarning
        class ReproducibilityWarning(UserWarning):
            pass
        
        fake_module.ReproducibilityWarning = ReproducibilityWarning
        
        # No-op functions that pyannote expects
        def fix_reproducibility(pipeline):
            """No-op: we want TF32 enabled for speed"""
            return pipeline
        
        def assert_googol_precision():
            """No-op: skip precision checks"""
            pass
        
        def assert_no_tf32():
            """No-op: we want TF32 enabled"""
            pass
        
        fake_module.fix_reproducibility = fix_reproducibility
        fake_module.assert_googol_precision = assert_googol_precision
        fake_module.assert_no_tf32 = assert_no_tf32
        
        # Pre-register the fake module
        sys.modules['pyannote.audio.utils.reproducibility'] = fake_module
        
        print("[PATCH] Patched pyannote reproducibility (TF32 will stay enabled)", flush=True)
        
    except Exception as e:
        print(f"[PATCH] Warning: Could not patch pyannote TF32: {e}", flush=True)


# Apply patches immediately when this module is imported
apply_all_patches()
