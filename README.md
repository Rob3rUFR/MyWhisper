# 🎤 MyWhisper - Transcription Audio avec IA

Conteneur Docker plug-and-play pour la transcription audio utilisant **Faster Whisper Large v3** avec diarisation des speakers via **Pyannote**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900)

## ✨ Fonctionnalités

- ✅ **Faster Whisper Large v3** - Transcription rapide et précise
- ✅ **Diarisation speakers** - Identification des intervenants (pyannote)
- ✅ **Multi-langue** - Détection automatique ou sélection manuelle
- ✅ **Export multi-format** - JSON, TXT, SRT, VTT
- ✅ **API OpenAI-compatible** - Intégration directe avec Open WebUI
- ✅ **Interface web** - Drag & drop moderne
- ✅ **GPU acceleration** - Optimisé pour RTX 3090

## 🚀 Quick Start

### Prérequis

- Docker Desktop avec WSL2
- NVIDIA Container Toolkit ([Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- GPU CUDA-compatible
- Token Hugging Face (pour la diarisation)

### Installation

```bash
# 1. Cloner le repo
git clone <repo_url>
cd whisper-stt

# 2. Configuration
copy env.example .env
# Éditer .env et ajouter votre HF_TOKEN

# 3. Build & Run
docker-compose up -d

# 4. Accès
# Interface: http://localhost:8000
# API: http://localhost:8000/v1/audio/transcriptions
```

### Obtenir un token Hugging Face

1. Créer un compte sur [huggingface.co](https://huggingface.co)
2. Aller dans Settings > Access Tokens
3. Créer un token avec accès en lecture
4. **Important**: Accepter les conditions d'utilisation de [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
5. Ajouter le token dans votre fichier `.env`

## 📖 Utilisation

### Interface Web

Accéder à `http://localhost:8000` pour l'interface drag & drop.

1. Glisser-déposer un fichier audio
2. Sélectionner les options (langue, format, diarisation)
3. Cliquer sur "Transcrire"
4. Télécharger ou copier le résultat

### API REST

#### Transcription (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=fr" \
  -F "response_format=json" \
  -F "diarize=true"
```

#### Réponse JSON

```json
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
      "speaker": "SPEAKER_00"
    }
  ]
}
```

#### Autres endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Interface web |
| `GET /health` | Health check (status GPU, modèles) |
| `GET /v1/models` | Liste des modèles disponibles |
| `POST /v1/audio/transcriptions` | Transcription OpenAI-compatible |
| `POST /transcribe` | Endpoint simplifié |

## ⚙️ Configuration Open WebUI

Pour utiliser MyWhisper comme moteur STT dans Open WebUI :

1. Aller dans **Settings > Audio > STT**
2. Configurer :
   - **Engine**: OpenAI
   - **Base URL**: `http://whisper-stt:8000/v1`
   - **API Key**: (laisser vide ou mettre une clé factice)
   - **Model**: `whisper-large-v3`

### Configuration réseau Docker

Si Open WebUI est dans un autre conteneur, ajouter les deux au même réseau :

```yaml
# Dans docker-compose.yml de MyWhisper
networks:
  - openwebui_network

networks:
  openwebui_network:
    external: true
```

## 🔧 Configuration avancée

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `WHISPER_MODEL` | `large-v3` | Modèle Whisper à utiliser |
| `DEVICE` | `cuda` | Device (cuda/cpu) |
| `COMPUTE_TYPE` | `float16` | Type de calcul (float16/int8) |
| `HF_TOKEN` | - | Token Hugging Face (requis pour diarisation) |
| `ENABLE_DIARIZATION` | `true` | Activer la diarisation |
| `MAX_FILE_SIZE` | `524288000` | Taille max fichier (500MB) |

### Optimisation GPU

Pour RTX 3090 (configuration optimale) :
```env
DEVICE=cuda
COMPUTE_TYPE=float16
```

Pour GPU avec moins de VRAM :
```env
COMPUTE_TYPE=int8
```

## 📁 Structure du projet

```
whisper-stt/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── env.example
├── .dockerignore
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + routes
│   ├── transcription.py     # Core STT logic
│   ├── diarization.py       # Speaker diarization
│   ├── utils.py             # Helpers
│   ├── config.py            # Configuration
│   │
│   └── static/
│       ├── index.html       # Interface web
│       ├── styles.css       # Styles
│       └── app.js           # Alpine.js app
│
├── uploads/                 # Fichiers temporaires (volume)
├── outputs/                 # Exports (volume)
└── models/                  # Modèles téléchargés (volume)
```

## 🐛 Troubleshooting

### GPU non détecté

```bash
# Vérifier NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Erreur CUDA out of memory

Réduire la précision dans `.env` :
```env
COMPUTE_TYPE=int8
```

### Diarisation échoue

1. Vérifier que `HF_TOKEN` est défini dans `.env`
2. Accepter les conditions sur [Hugging Face](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Vérifier les logs : `docker-compose logs -f`

### Port 8000 déjà utilisé

Changer le port dans `docker-compose.yml` :
```yaml
ports:
  - "8001:8000"
```

## 📊 Performance

**Hardware cible** : RTX 3090 (24GB VRAM)

| Métrique | Valeur |
|----------|--------|
| Chargement modèle | < 10s |
| Transcription 1min audio | < 5s |
| VRAM usage | < 6GB |
| Diarisation overhead | +30% temps max |

## 🔄 Commandes utiles

```bash
# Build
docker-compose build

# Démarrer
docker-compose up -d

# Logs en temps réel
docker-compose logs -f

# Arrêter
docker-compose down

# Vérifier GPU dans le conteneur
docker exec whisper-stt nvidia-smi

# Shell dans le conteneur
docker exec -it whisper-stt bash

# Supprimer cache modèles
docker exec whisper-stt rm -rf /app/models/*
docker-compose restart
```

## 📜 Licence

MIT License

---

**Créé avec ❤️ pour une transcription audio simple et efficace**
