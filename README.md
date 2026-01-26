# 🎤 MyWhisper - Transcription Audio avec IA

Conteneur Docker plug-and-play pour la transcription audio utilisant **Faster Whisper Large v3 Turbo** et post-traitement via **Ollama**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900)

## ✨ Fonctionnalités

### Transcription
- 🎯 **Faster Whisper Large v3 Turbo** - Transcription rapide et précise
- 🌍 **Multi-langue** - Détection automatique ou sélection manuelle (50+ langues)
- 📄 **Export multi-format** - JSON, TXT, SRT, VTT

### Dictée en temps réel
- 🎙️ **Enregistrement micro** - Dictée vocale directement depuis le navigateur
- ⚡ **Transcription live** - Résultats en temps réel pendant l'enregistrement
- 🔇 **Détection silence** - Arrêt automatique après 10s de silence
- 📊 **VU-mètre** - Indicateur visuel du niveau audio

### Post-traitement IA (Ollama)
- 🤖 **Intégration Ollama** - Connexion à votre instance Ollama locale
- 📝 **Prompts personnalisés** - Créez vos propres prompts de reformulation
- 📋 **Copie formatée** - Export optimisé pour Word/Outlook avec mise en forme

### Interface & UX
- 🖥️ **Interface web moderne** - Drag & drop, dark mode
- 📈 **Progression détaillée** - Suivi en temps réel de chaque étape
- 💾 **Sauvegarde automatique** - Export direct vers un dossier de votre choix
- 🔌 **API OpenAI-compatible** - Intégration directe avec Open WebUI
- 🔄 **Persistance de l'état** - L'interface conserve son état après refresh
- 📂 **Historique des transcriptions** - Gestion et téléchargement des anciennes transcriptions

### Performance
- 🚀 **GPU acceleration** - Optimisé CUDA avec TF32
- 💪 **Support RTX 5090** - Compatible avec les derniers GPU NVIDIA
- ⚡ **VAD intégré** - Filtrage automatique des silences
- 🔁 **Récupération auto** - Reprise après perte de connexion

## 🚀 Quick Start

### Prérequis

- Docker Desktop avec WSL2
- NVIDIA Container Toolkit ([Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- GPU CUDA-compatible (RTX 3000/4000/5000 series)
- (Optionnel) Ollama pour le post-traitement LLM

### Installation

```bash
# 1. Cloner le repo
git clone https://github.com/VOTRE_USERNAME/MyWhisper.git
cd MyWhisper

# 2. Configuration
copy env.example .env
# Éditer .env si nécessaire

# 3. Build & Run
docker-compose up -d

# 4. Accès
# Interface: http://localhost:8000
# API: http://localhost:8000/v1/audio/transcriptions
```

## 📖 Utilisation

### Interface Web

Accéder à `http://localhost:8000` pour l'interface complète.

#### Onglet Fichier
1. Glisser-déposer un fichier audio/vidéo
2. Sélectionner les options (langue, format)
3. Activer la sauvegarde automatique si souhaité
4. Cliquer sur "Transcrire"
5. Retraiter avec l'IA (Ollama) si configuré

#### Onglet Dictée
1. Sélectionner la langue
2. Cliquer sur "Démarrer"
3. Parler dans le micro
4. La transcription apparaît en temps réel
5. Arrêt automatique après 10s de silence ou clic sur "Arrêter"

#### Onglet Historique
1. Voir toutes les transcriptions passées
2. Télécharger dans différents formats (Texte, JSON, SRT, VTT)
3. Visualiser ou supprimer une transcription
4. Configurer la durée de rétention dans les paramètres

#### Onglet Paramètres
1. Configurer la durée de conservation de l'historique (1-365 jours)
2. Configurer l'URL Ollama (ex: `http://host.docker.internal:11434`)
3. Sélectionner un modèle LLM
4. Créer des prompts personnalisés avec `{text}` comme placeholder

### API REST

#### Transcription (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=fr" \
  -F "response_format=json"
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
      "text": "Bonjour, aujourd'hui..."
    }
  ]
}
```

#### Endpoints disponibles

| Endpoint | Description |
|----------|-------------|
| `GET /` | Interface web |
| `GET /health` | Health check (status GPU, modèles) |
| `GET /v1/models` | Liste des modèles disponibles |
| `POST /v1/audio/transcriptions` | Transcription OpenAI-compatible |
| `POST /v1/audio/transcriptions/stream` | Transcription avec SSE (évite timeout) |
| `POST /transcribe` | Endpoint simplifié |
| `GET /history` | Liste des transcriptions (pagination) |
| `GET /history/{id}` | Détails d'une transcription |
| `GET /history/{id}/download` | Télécharger (format: text/json/srt/vtt) |
| `DELETE /history/{id}` | Supprimer une transcription |
| `GET /result/{client_id}` | Récupérer résultat après déconnexion |

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
| `WHISPER_MODEL` | `large-v3-turbo` | Modèle Whisper à utiliser |
| `DEVICE` | `cuda` | Device (cuda/cpu) |
| `COMPUTE_TYPE` | `float16` | Type de calcul (float16/int8) |
| `MAX_FILE_SIZE` | `524288000` | Taille max fichier (500MB) |
| `OLLAMA_URL` | - | URL de l'instance Ollama |
| `HISTORY_RETENTION_DAYS` | `90` | Durée conservation historique (jours) |

### Modèles Whisper disponibles

| Modèle | Vitesse | Qualité | VRAM |
|--------|---------|---------|------|
| `large-v3` | Référence | Meilleure | ~10GB |
| `large-v3-turbo` | ~2-3x plus rapide | Légèrement inférieure | ~6GB |
| `medium` | Rapide | Bonne | ~5GB |
| `small` | Très rapide | Correcte | ~2GB |

### Optimisation GPU

Pour RTX 4090/5090 (configuration optimale) :
```env
DEVICE=cuda
COMPUTE_TYPE=float16
```

Pour GPU avec moins de VRAM (8-12GB) :
```env
COMPUTE_TYPE=int8
```

## 📁 Structure du projet

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
│   ├── patches.py           # Compatibility patches (PyTorch)
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
└── models/                  # Modèles téléchargés (volume, gitignore)
```

## 🐛 Troubleshooting

### GPU non détecté

```bash
# Vérifier NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

### Erreur CUDA out of memory

Réduire la précision dans `.env` :
```env
COMPUTE_TYPE=int8
```

Ou utiliser un modèle plus petit :
```env
WHISPER_MODEL=medium
```

### Erreur "weights_only" PyTorch

Le patch est automatiquement appliqué. Si problème, ajouter dans `docker-compose.yml` :
```yaml
environment:
  - TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
```

### Port 8000 déjà utilisé

Changer le port dans `docker-compose.yml` :
```yaml
ports:
  - "8001:8000"
```

### Ollama non connecté

1. Vérifier qu'Ollama est lancé : `ollama serve`
2. **Important** : Ollama doit écouter sur `0.0.0.0` pour être accessible depuis Docker
   - Définir `OLLAMA_HOST=0.0.0.0:11434` avant de lancer Ollama
3. Utiliser `http://host.docker.internal:11434` dans `.env`

## 📊 Performance

**Hardware testé** : RTX 5090

| Métrique | Valeur |
|----------|--------|
| Chargement modèle | ~5s |
| Transcription 1min audio | < 3s |
| Transcription 1h audio | ~5min |
| VRAM usage | ~4-6GB |

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

# Reconstruire et relancer
docker-compose up -d --build

# Vérifier GPU dans le conteneur
docker exec whisper-stt nvidia-smi

# Shell dans le conteneur
docker exec -it whisper-stt bash

# Supprimer cache modèles
docker exec whisper-stt rm -rf /app/models/*
docker-compose restart
```

## 🆕 Changelog

### v2.0.0 (Janvier 2026)
- ❌ **Suppression de la diarisation** - Fonctionnalité retirée (non fiable)
- ✅ **Modèle large-v3-turbo** - Transcription 2-3x plus rapide
- ✅ **Interface simplifiée** - Plus d'options de diarisation
- ✅ **Image Docker allégée** - Suppression des dépendances NeMo

### v1.2.0 (Janvier 2026)
- ✅ **Historique des transcriptions** - Conservation avec durée configurable
- ✅ **Persistance de l'état** - L'interface conserve son état après refresh
- ✅ **Récupération automatique** - Reprise des résultats après perte de connexion

### v1.0.0
- ✅ Transcription Faster Whisper Large v3
- ✅ Interface web moderne (Alpine.js)
- ✅ API OpenAI-compatible
- ✅ Dictée en temps réel avec détection silence
- ✅ Intégration Ollama pour post-traitement LLM
- ✅ Support RTX 5090 (CUDA 12.4, PyTorch 2.9)

## 📜 Licence

MIT License

---

**Créé avec ❤️ pour une transcription audio simple et efficace**
