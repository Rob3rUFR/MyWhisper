# PRD - MyWhisper

## 1. Vue d'ensemble

**Objectif** : Créer un conteneur Docker plug-and-play pour la transcription audio (STT) basé sur Faster Whisper Large v3, avec interface web minimaliste et API OpenAI-compatible pour intégration Open WebUI.

**Environnement cible** : Windows Server avec dual RTX 3090, 96GB RAM, support CUDA 12.x

---

## 2. Stack Technique

```yaml
Base Image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
Python: 3.11
STT Engine: faster-whisper (large-v3)
Diarization: pyannote.audio 3.1+
API Framework: FastAPI 0.110+
Web Server: Uvicorn
Frontend: HTML5 + Vanilla JS (Alpine.js pour réactivité)
File Storage: Volume Docker monté
GPU Acceleration: CTranslate2 (automatique)
```

---

## 3. Architecture

### 3.1 Structure des dossiers

```
whisper-stt/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── README.md
│
├── app/
│   ├── main.py                 # FastAPI app + routes
│   ├── transcription.py        # Core STT logic
│   ├── diarization.py          # Speaker diarization
│   ├── utils.py                # Helpers (file validation, etc.)
│   ├── config.py               # Configuration management
│   │
│   ├── static/
│   │   ├── index.html          # Interface upload
│   │   ├── styles.css          # Styling minimal
│   │   └── app.js              # Upload logic + API calls
│   │
│   └── models/                 # Downloaded models (volume)
│
├── uploads/                    # Fichiers audio temporaires (volume)
└── outputs/                    # Transcriptions exportées (volume)
```

---

## 4. Spécifications Fonctionnelles

### 4.1 Interface Web (`/`)

**Features** :
- [ ] Drag & drop zone pour fichiers audio
- [ ] Upload button classique (fallback)
- [ ] Formats acceptés : MP3, WAV, M4A, FLAC, OGG, WEBM
- [ ] Taille max : 500MB par fichier
- [ ] Barre de progression upload
- [ ] Options visibles :
  - [ ] Détection langue automatique (ON par défaut)
  - [ ] Diarisation speakers (checkbox)
  - [ ] Format export : TXT / SRT / VTT
- [ ] Affichage résultat temps réel
- [ ] Bouton download transcription

**UX** :
- Design minimal, dark mode par défaut
- Feedback visuel immédiat (spinner, success/error states)
- Pas de login requis

---

### 4.2 API OpenAI-Compatible

**Endpoint principal** : `POST /v1/audio/transcriptions`

**Conformité OpenAI Whisper API** :
```python
# Request (multipart/form-data)
{
  "file": <audio_file>,           # Required
  "model": "whisper-large-v3",    # Ignoré (toujours large-v3)
  "language": "fr",                # Optional (auto-detect si absent)
  "response_format": "json",       # json | text | srt | vtt
  "timestamp_granularities": [],   # Ignoré pour l'instant
  "diarize": false                 # Extension custom (bool)
}

# Response (format json)
{
  "text": "Transcription complète...",
  "language": "fr",
  "duration": 145.2,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Bonjour, aujourd'hui...",
      "speaker": "SPEAKER_00"  # Si diarize=true
    }
  ]
}
```

**Endpoints additionnels** :
- `GET /health` - Health check (pour Open WebUI)
- `GET /models` - Liste des modèles disponibles
- `POST /transcribe` - Alias simplifié (même logique que `/v1/audio/transcriptions`)

---

## 5. Spécifications Techniques

### 5.1 Transcription Engine (`app/transcription.py`)

**Classe principale** : `WhisperTranscriber`

```python
class WhisperTranscriber:
    def __init__(self, model_name="large-v3", device="cuda", compute_type="float16"):
        """
        Initialise Faster Whisper avec config optimale pour RTX 3090
        - device: cuda | cpu
        - compute_type: float16 (meilleur ratio speed/quality sur 3090)
        - num_workers: 4 (optimal pour dual 3090)
        """
        
    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        task: str = "transcribe",
        vad_filter: bool = True
    ) -> dict:
        """
        Transcription principale
        - Auto-détection langue si language=None
        - VAD filter pour enlever silences
        - Retourne segments avec timestamps
        """
```

**Configuration Faster Whisper** :
```python
model = WhisperModel(
    model_size_or_path="large-v3",
    device="cuda",
    compute_type="float16",
    num_workers=4,
    download_root="./app/models/whisper"
)
```

---

### 5.2 Diarization (`app/diarization.py`)

**Classe** : `SpeakerDiarizer`

```python
from pyannote.audio import Pipeline

class SpeakerDiarizer:
    def __init__(self, hf_token: str = None):
        """
        Charge pyannote/speaker-diarization-3.1
        - Nécessite token Hugging Face (env var HF_TOKEN)
        - Cache modèle dans ./app/models/pyannote
        """
        
    def diarize(self, audio_path: str) -> dict:
        """
        Retourne speaker timeline
        Format: [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}]
        """
        
    def merge_with_transcription(
        self,
        transcription_segments: list,
        diarization_timeline: list
    ) -> list:
        """
        Fusionne timestamps Whisper + labels speakers
        Logique: assigner speaker dominant sur chaque segment Whisper
        """
```

**Note importante** : Diarization requiert `HF_TOKEN` (Hugging Face) pour télécharger le modèle pyannote. À documenter dans README.

---

### 5.3 API FastAPI (`app/main.py`)

**Routes** :

```python
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Whisper STT API", version="1.0.0")

# Serve interface web
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    """Serve index.html"""
    
@app.get("/health")
async def health_check():
    """Retourne status GPU + modèles chargés"""
    
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile,
    language: str = Form(None),
    response_format: str = Form("json"),
    diarize: bool = Form(False)
):
    """
    OpenAI-compatible transcription endpoint
    1. Valider fichier (format, taille)
    2. Sauver temporairement dans /uploads
    3. Transcription Whisper
    4. Diarization si demandé
    5. Formater réponse selon response_format
    6. Cleanup fichier temporaire
    """
    
@app.post("/transcribe")
async def simple_transcribe(file: UploadFile):
    """Alias simplifié, retourne toujours JSON"""
```

---

### 5.4 Configuration (`app/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Modèle
    WHISPER_MODEL: str = "large-v3"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"
    
    # Diarization
    ENABLE_DIARIZATION: bool = True
    HF_TOKEN: str = ""  # Obligatoire si diarization activée
    
    # Limites
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_FORMATS: list = ["mp3", "wav", "m4a", "flac", "ogg", "webm"]
    
    # Serveur
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MODEL_DIR: str = "./app/models"
    
    class Config:
        env_file = ".env"
```

---

## 6. Docker Configuration

### 6.1 Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Create directories
RUN mkdir -p /app/uploads /app/outputs /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 6.2 docker-compose.yml

```yaml
version: '3.8'

services:
  whisper-stt:
    build: .
    container_name: whisper-stt
    
    ports:
      - "8000:8000"
    
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./app/models:/app/models
    
    environment:
      - WHISPER_MODEL=large-v3
      - DEVICE=cuda
      - COMPUTE_TYPE=float16
      - HF_TOKEN=${HF_TOKEN}  # Depuis .env
      - ENABLE_DIARIZATION=true
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Utilise 1 GPU (auto-select)
              capabilities: [gpu]
    
    restart: unless-stopped
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### 6.3 requirements.txt

```txt
# Core
fastapi==0.110.0
uvicorn[standard]==0.27.1
python-multipart==0.0.9
pydantic-settings==2.2.1

# STT Engine
faster-whisper==1.0.0
torch==2.2.0
torchaudio==2.2.0

# Diarization
pyannote.audio==3.1.1
speechbrain==1.0.0

# Utils
pydub==0.25.1
python-dotenv==1.0.1
aiofiles==23.2.1
```

---

## 7. Interface Web (`app/static/`)

### 7.1 index.html (Structure)

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Whisper STT</title>
    <link rel="stylesheet" href="/static/styles.css">
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
</head>
<body x-data="sttApp()">
    <div class="container">
        <h1>🎤 Transcription Audio</h1>
        
        <!-- Drop Zone -->
        <div class="dropzone" 
             @drop.prevent="handleDrop($event)"
             @dragover.prevent
             @dragenter.prevent="dragActive = true"
             @dragleave.prevent="dragActive = false"
             :class="{ 'drag-active': dragActive }">
            
            <input type="file" 
                   id="fileInput" 
                   accept=".mp3,.wav,.m4a,.flac,.ogg,.webm"
                   @change="handleFileSelect($event)"
                   hidden>
            
            <label for="fileInput" class="upload-label">
                📂 Glissez un fichier ou cliquez ici
            </label>
        </div>
        
        <!-- Options -->
        <div class="options">
            <label>
                <input type="checkbox" x-model="options.diarize">
                Diarisation (speakers)
            </label>
            
            <select x-model="options.format">
                <option value="json">JSON</option>
                <option value="text">TXT</option>
                <option value="srt">SRT</option>
                <option value="vtt">VTT</option>
            </select>
        </div>
        
        <!-- Progress -->
        <div x-show="uploading" class="progress">
            <div class="progress-bar" :style="`width: ${progress}%`"></div>
        </div>
        
        <!-- Results -->
        <div x-show="result" class="result">
            <h3>📝 Transcription</h3>
            <pre x-text="result"></pre>
            <button @click="downloadResult()">💾 Télécharger</button>
        </div>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>
```

---

### 7.2 app.js (Logique Alpine.js)

```javascript
function sttApp() {
    return {
        dragActive: false,
        uploading: false,
        progress: 0,
        result: null,
        options: {
            diarize: false,
            format: 'json'
        },
        
        handleDrop(event) {
            this.dragActive = false;
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                this.uploadFile(files[0]);
            }
        },
        
        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                this.uploadFile(file);
            }
        },
        
        async uploadFile(file) {
            this.uploading = true;
            this.progress = 0;
            this.result = null;
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('response_format', this.options.format);
            formData.append('diarize', this.options.diarize);
            
            try {
                const response = await fetch('/v1/audio/transcriptions', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Transcription failed');
                
                const data = await response.json();
                this.result = this.options.format === 'json' 
                    ? JSON.stringify(data, null, 2)
                    : data.text;
                    
            } catch (error) {
                alert('Erreur: ' + error.message);
            } finally {
                this.uploading = false;
                this.progress = 100;
            }
        },
        
        downloadResult() {
            const blob = new Blob([this.result], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcription.${this.options.format}`;
            a.click();
        }
    }
}
```

---

### 7.3 styles.css (Minimal Dark Theme)

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}

.container {
    max-width: 800px;
    width: 100%;
}

h1 {
    text-align: center;
    margin-bottom: 2rem;
    font-size: 2rem;
    font-weight: 600;
}

.dropzone {
    border: 2px dashed #333;
    border-radius: 12px;
    padding: 4rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    background: #1a1a1a;
}

.dropzone:hover,
.dropzone.drag-active {
    border-color: #4a9eff;
    background: #1f1f1f;
}

.upload-label {
    cursor: pointer;
    font-size: 1.2rem;
    color: #888;
}

.options {
    margin: 2rem 0;
    display: flex;
    gap: 2rem;
    justify-content: center;
    align-items: center;
}

.options label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.options select {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #e0e0e0;
    padding: 0.5rem 1rem;
    border-radius: 6px;
}

.progress {
    width: 100%;
    height: 4px;
    background: #1a1a1a;
    border-radius: 2px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #4a9eff, #7b68ee);
    transition: width 0.3s ease;
}

.result {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 2rem;
    margin-top: 2rem;
}

.result h3 {
    margin-bottom: 1rem;
}

.result pre {
    background: #0f0f0f;
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
    font-size: 0.9rem;
    line-height: 1.6;
}

button {
    background: #4a9eff;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    margin-top: 1rem;
    transition: background 0.2s;
}

button:hover {
    background: #3a7fd5;
}
```

---

## 8. Intégration Open WebUI

**Configuration dans Open WebUI** :

```yaml
# Settings > Audio > STT
STT Engine: OpenAI
API Base URL: http://whisper-stt:8000/v1
API Key: (laisser vide ou dummy key)
Model: whisper-large-v3
```

**Network Docker** (ajouter au docker-compose.yml) :

```yaml
networks:
  - openwebui_network

networks:
  openwebui_network:
    external: true
```

---

## 9. .env.example

```env
# Hugging Face Token (requis pour diarization)
HF_TOKEN=your_hf_token_here

# Model Configuration
WHISPER_MODEL=large-v3
DEVICE=cuda
COMPUTE_TYPE=float16

# Diarization
ENABLE_DIARIZATION=true

# Limits
MAX_FILE_SIZE=524288000
```

---

## 10. Critères d'Acceptance

### Must Have ✅
- [ ] Container build et démarre sans erreur sur Windows + Docker Desktop
- [ ] GPU détecté et utilisé (vérifiable dans logs)
- [ ] Interface web accessible sur `http://localhost:8000`
- [ ] Upload fichier MP3/WAV fonctionne
- [ ] Transcription retourne texte correct (test FR + EN)
- [ ] API `/v1/audio/transcriptions` compatible OpenAI
- [ ] Diarization retourne speaker labels corrects
- [ ] Export TXT téléchargeable
- [ ] Health check `/health` retourne 200

### Nice to Have 🎯
- [ ] Support SRT/VTT avec timestamps
- [ ] Détection langue affichée dans UI
- [ ] Logs structurés (JSON)
- [ ] Métriques GPU (VRAM usage)
- [ ] Queue system pour batch processing

---

## 11. Performance Targets

**Hardware cible** : RTX 3090 (24GB VRAM)

- Chargement modèle : < 10s
- Transcription 1min audio : < 5s
- VRAM usage : < 6GB (large-v3 + diarization)
- Diarization overhead : +30% temps total max

---

## 12. README.md (Template)

````markdown
# 🎤 Whisper STT Docker Container

Conteneur Docker plug-and-play pour transcription audio avec Faster Whisper Large v3 et diarization.

## Quick Start

```bash
# 1. Clone repo
git clone <repo_url>
cd whisper-stt

# 2. Configuration
cp .env.example .env
# Éditer .env et ajouter votre HF_TOKEN

# 3. Build & Run
docker-compose up -d

# 4. Accès
Interface: http://localhost:8000
API: http://localhost:8000/v1/audio/transcriptions
```

## Prérequis

- Docker Desktop avec WSL2
- NVIDIA Container Toolkit
- GPU CUDA-compatible
- Token Hugging Face (pour diarization)

## Configuration Open WebUI

Settings > Audio > STT:
- Engine: OpenAI
- Base URL: `http://whisper-stt:8000/v1`
- Model: `whisper-large-v3`

## API Usage

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "response_format=json"
```

## Features

✅ Faster Whisper Large v3  
✅ Diarization speakers (pyannote)  
✅ Multi-langue (auto-detect)  
✅ Export TXT/SRT/VTT  
✅ API OpenAI-compatible  
✅ Interface web drag & drop  
✅ GPU acceleration  

## Troubleshooting

**GPU not detected**
```bash
# Vérifier NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Diarization fails**
- Vérifier que HF_TOKEN est défini dans .env
- Accepter les conditions d'utilisation sur https://huggingface.co/pyannote/speaker-diarization-3.1

## License

MIT
````

---

## 13. Notes d'implémentation

**Optimisations GPU** :
- `compute_type="float16"` optimal pour 3090 (balance speed/quality)
- `num_workers=4` pour parallélisation
- Batch size=1 par défaut (temps réel prioritaire)

**Gestion mémoire** :
- Cleanup automatique fichiers uploads après transcription
- Cache modèles sur volume persistant
- Limite 500MB par fichier (configurable)

**Sécurité** :
- Validation stricte formats fichiers
- Sanitization noms fichiers
- Pas d'auth (usage local trusted)

**Error Handling** :
- Timeout transcription : 5 minutes max
- Retry logic pour download modèles
- Logs détaillés avec timestamps

---

## 14. Fichiers supplémentaires recommandés

### 14.1 .dockerignore

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git/
.gitignore
*.md
!README.md
uploads/*
outputs/*
.env
.DS_Store
```

---

### 14.2 app/utils.py (Helpers)

```python
import os
from pathlib import Path
from typing import Optional
import hashlib

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"}

def validate_audio_file(filename: str, max_size: int) -> tuple[bool, Optional[str]]:
    """
    Valide un fichier audio
    Returns: (is_valid, error_message)
    """
    # Check extension
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Format non supporté. Formats acceptés: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None

def sanitize_filename(filename: str) -> str:
    """
    Sécurise un nom de fichier
    """
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Generate safe name with hash
    name, ext = os.path.splitext(filename)
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    safe_name = f"{hash_suffix}_{name[:50]}{ext}"
    
    return safe_name

def format_timestamp(seconds: float) -> str:
    """
    Formate timestamp en HH:MM:SS,mmm (format SRT)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def segments_to_srt(segments: list) -> str:
    """
    Convertit segments en format SRT
    """
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        text = seg['text'].strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line
    
    return "\n".join(srt_lines)

def segments_to_vtt(segments: list) -> str:
    """
    Convertit segments en format VTT
    """
    vtt_lines = ["WEBVTT", ""]
    
    for seg in segments:
        start = format_timestamp(seg['start']).replace(',', '.')
        end = format_timestamp(seg['end']).replace(',', '.')
        text = seg['text'].strip()
        
        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(text)
        vtt_lines.append("")
    
    return "\n".join(vtt_lines)
```

---

## 15. Tests recommandés

### 15.1 Test files à inclure

```
tests/
├── audio_samples/
│   ├── test_fr_mono.mp3      # Fichier français 30s
│   ├── test_en_stereo.wav    # Fichier anglais 1min
│   └── test_multi_speaker.m4a # 2+ speakers
│
└── test_api.py
```

### 15.2 test_api.py (Tests basiques)

```python
import pytest
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "gpu_available" in data

def test_transcription_json():
    audio_file = Path("tests/audio_samples/test_fr_mono.mp3")
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        data = {"response_format": "json"}
        response = requests.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data
        )
    
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert len(result["text"]) > 0

def test_diarization():
    audio_file = Path("tests/audio_samples/test_multi_speaker.m4a")
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        data = {"diarize": "true"}
        response = requests.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data
        )
    
    assert response.status_code == 200
    result = response.json()
    
    # Check speaker labels
    speakers = set()
    for seg in result.get("segments", []):
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    
    assert len(speakers) >= 2  # Au moins 2 speakers détectés
```

---

## 16. Déploiement & Maintenance

### 16.1 Mise à jour des modèles

```bash
# Entrer dans le container
docker exec -it whisper-stt bash

# Supprimer cache modèles
rm -rf /app/models/*

# Redémarrer pour re-télécharger
docker-compose restart
```

### 16.2 Monitoring

**Logs Docker** :
```bash
docker-compose logs -f whisper-stt
```

**GPU Usage** :
```bash
docker exec whisper-stt nvidia-smi
```

### 16.3 Backup recommandé

**Volumes à sauvegarder** :
- `./app/models/` (modèles téléchargés ~10GB)
- `.env` (configuration)

**Volumes jetables** :
- `./uploads/` (temporaire)
- `./outputs/` (récupérables)

---

## 17. Roadmap Future

**Phase 2 (optionnel)** :
- [ ] Streaming audio support (websocket)
- [ ] Batch processing UI
- [ ] Multi-GPU load balancing
- [ ] Redis queue pour jobs
- [ ] Webhooks callback
- [ ] Support vidéo (extract audio)
- [ ] Custom vocabulary/glossary
- [ ] Translation task support

---

**FIN DU PRD** - Version 1.0 - Prêt pour implémentation Cursor 🚀

---

## Annexe A - Commandes Quick Reference

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down

# GPU Check
docker exec whisper-stt nvidia-smi

# Shell
docker exec -it whisper-stt bash

# API Test
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test.mp3" \
  -F "diarize=true"
```

---

## Annexe B - Troubleshooting Guide

| Problème | Cause probable | Solution |
|----------|----------------|----------|
| `RuntimeError: CUDA out of memory` | VRAM insuffisante | Utiliser `compute_type="int8"` ou modèle plus petit |
| `HuggingFace token error` | HF_TOKEN manquant/invalide | Vérifier .env et accepter conditions pyannote |
| `FileNotFoundError: ffmpeg` | FFmpeg non installé | Vérifier Dockerfile apt install ffmpeg |
| `GPU not available` | NVIDIA runtime manquant | Installer NVIDIA Container Toolkit |
| `Port 8000 already in use` | Conflit port | Changer port dans docker-compose.yml |
| `Model download stuck` | Réseau/proxy | Vérifier connexion internet, retry |

---

**Document maintenu par** : Louis-Adrien  
**Dernière mise à jour** : Janvier 2026  
**Version** : 1.0.0
