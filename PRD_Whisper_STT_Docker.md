# PRD - MyWhisper

## 1. Vue d'ensemble

**Objectif** : Créer un conteneur Docker plug-and-play pour la transcription audio (STT) basé sur Faster Whisper Large v3 Turbo, avec interface web moderne et API OpenAI-compatible pour intégration Open WebUI.

**Environnement cible** : Windows/Linux avec GPU NVIDIA (RTX 3000/4000/5000 series), support CUDA 12.x

---

## 2. Stack Technique

```yaml
Base Image: nvidia/nemo:25.09.01
Python: 3.12
STT Engine: faster-whisper (large-v3-turbo)
API Framework: FastAPI 0.110+
Web Server: Uvicorn
Frontend: HTML5 + Alpine.js
File Storage: Volume Docker monté
GPU Acceleration: CTranslate2 (automatique)
LLM Post-processing: Ollama (optionnel)
```

---

## 3. Architecture

### 3.1 Structure des dossiers

```
MyWhisper/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── env.example
├── .dockerignore
├── .gitignore
├── README.md
├── PRD_Whisper_STT_Docker.md
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + routes
│   ├── transcription.py     # Core STT logic (Faster Whisper)
│   ├── history.py           # Gestion historique (SQLite)
│   ├── patches.py           # Compatibility patches
│   ├── utils.py             # Helpers (formats, validation)
│   ├── config.py            # Configuration (Pydantic)
│   │
│   └── static/
│       ├── index.html       # Interface web (4 onglets)
│       ├── styles.css       # Dark theme moderne
│       └── app.js           # Alpine.js app
│
├── uploads/                 # Fichiers temporaires (volume)
├── outputs/                 # Exports (volume)
└── models/                  # Modèles téléchargés (volume)
```

---

## 4. Spécifications Fonctionnelles

### 4.1 Interface Web (`/`)

**Features** :
- [x] Drag & drop zone pour fichiers audio
- [x] Upload button classique (fallback)
- [x] Formats acceptés : MP3, WAV, M4A, FLAC, OGG, WEBM, MP4
- [x] Taille max : 500MB par fichier
- [x] Barre de progression avec étapes
- [x] Options visibles :
  - [x] Sélection langue (auto-détection par défaut)
  - [x] Format export : JSON / TXT / SRT / VTT
- [x] Affichage résultat avec mise en forme
- [x] Bouton download transcription
- [x] Dictée en temps réel (microphone)
- [x] Historique des transcriptions
- [x] Intégration Ollama pour post-traitement

**UX** :
- Design moderne, dark mode
- 4 onglets : Fichier, Dictée, Historique, Paramètres
- Feedback visuel immédiat (spinner, success/error states)
- Persistance de l'état après refresh (sessionStorage)
- Pas de login requis

---

### 4.2 API OpenAI-Compatible

**Endpoint principal** : `POST /v1/audio/transcriptions`

**Conformité OpenAI Whisper API** :
```python
# Request (multipart/form-data)
{
  "file": <audio_file>,           # Required
  "model": "whisper-large-v3",    # Ignoré (toujours large-v3-turbo)
  "language": "fr",               # Optional (auto-detect si absent)
  "response_format": "json"       # json | text | srt | vtt
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
      "text": "Bonjour, aujourd'hui..."
    }
  ]
}
```

**Endpoints** :
- `GET /health` - Health check (GPU, modèle)
- `GET /v1/models` - Liste des modèles
- `POST /v1/audio/transcriptions` - Transcription OpenAI-compatible
- `POST /v1/audio/transcriptions/stream` - Transcription avec SSE
- `POST /transcribe` - Alias simplifié
- `GET /history` - Liste des transcriptions
- `GET /history/{id}` - Détails transcription
- `DELETE /history/{id}` - Supprimer transcription
- `GET /ollama/status` - Status Ollama
- `POST /ollama/generate` - Proxy Ollama

---

## 5. Spécifications Techniques

### 5.1 Transcription Engine (`app/transcription.py`)

**Classe principale** : `WhisperTranscriber`

```python
class WhisperTranscriber:
    def __init__(self, model_name="large-v3-turbo", device="cuda", compute_type="float16"):
        """
        Initialise Faster Whisper avec config optimale
        - device: cuda | cpu
        - compute_type: float16 (meilleur ratio speed/quality)
        - num_workers: 4 (optimal pour parallélisation)
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

---

### 5.2 Configuration (`app/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Modèle
    WHISPER_MODEL: str = "large-v3-turbo"
    DEVICE: str = "cuda"
    COMPUTE_TYPE: str = "float16"
    NUM_WORKERS: int = 4
    
    # Limites
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_FORMATS: list = ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
    
    # Serveur
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MODEL_DIR: str = "./models"
    
    # Ollama
    OLLAMA_URL: str = ""
    OLLAMA_MODEL: str = ""
    
    # History
    HISTORY_RETENTION_DAYS: int = 90
    
    class Config:
        env_file = ".env"
```

---

## 6. Docker Configuration

### 6.1 Dockerfile

```dockerfile
FROM nvidia/nemo:25.09.01

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /app/uploads /app/outputs /app/models /app/app/static

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY app/*.py ./app/
COPY app/static/ ./app/static/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 6.2 docker-compose.yml

```yaml
services:
  whisper-stt:
    build: .
    container_name: whisper-stt
    
    ports:
      - "8000:8000"
    
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./models:/app/models
    
    env_file:
      - .env
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    restart: unless-stopped
    
    networks:
      - whisper-network

networks:
  whisper-network:
    driver: bridge
```

---

## 7. Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Modèle Whisper |
| `DEVICE` | `cuda` | Device (cuda/cpu) |
| `COMPUTE_TYPE` | `float16` | Type de calcul |
| `MAX_FILE_SIZE` | `524288000` | Taille max (500MB) |
| `OLLAMA_URL` | - | URL Ollama |
| `OLLAMA_MODEL` | - | Modèle LLM |
| `HISTORY_RETENTION_DAYS` | `90` | Rétention historique |

---

## 8. Performance Targets

**Hardware testé** : RTX 5090

| Métrique | Valeur |
|----------|--------|
| Chargement modèle | < 5s |
| Transcription 1min audio | < 3s |
| Transcription 1h audio | < 5min |
| VRAM usage | 4-6GB |

---

## 9. Critères d'Acceptance

### Must Have ✅
- [x] Container build et démarre sans erreur
- [x] GPU détecté et utilisé
- [x] Interface web accessible sur `http://localhost:8000`
- [x] Upload fichier MP3/WAV fonctionne
- [x] Transcription retourne texte correct
- [x] API `/v1/audio/transcriptions` compatible OpenAI
- [x] Export TXT/SRT/VTT téléchargeable
- [x] Health check `/health` retourne 200
- [x] Dictée temps réel fonctionne
- [x] Historique des transcriptions
- [x] Intégration Ollama

---

## 10. Changelog

### v2.0.0 (Janvier 2026)
- ❌ Suppression de la diarisation (non fiable)
- ✅ Modèle large-v3-turbo par défaut
- ✅ Interface simplifiée
- ✅ Image Docker allégée

### v1.0.0
- ✅ Transcription Faster Whisper
- ✅ Interface web moderne
- ✅ API OpenAI-compatible
- ✅ Dictée temps réel
- ✅ Intégration Ollama

---

**Document maintenu par** : Louis-Adrien  
**Dernière mise à jour** : Janvier 2026  
**Version** : 2.0.0
