/**
 * MyWhisper - Speech to Text Application
 * Alpine.js application for handling audio transcription
 */

function sttApp() {
    return {
        // State
        activeTab: 'file',
        
        // Ollama settings
        ollamaSettings: {
            url: 'http://localhost:11434',
            model: ''
        },
        ollamaModels: [],
        ollamaConnected: false,
        ollamaStatus: '',
        ollamaLoading: false,
        llmProcessing: false,
        llmResult: '',
        llmLastPrompt: '',
        llmCopied: false,
        
        // Custom prompts
        customPrompts: [
            { name: 'Corriger', content: 'Corrige les erreurs de transcription et améliore la ponctuation du texte suivant, sans changer le sens:\n\n{text}' },
            { name: 'Résumer', content: 'Résume le texte suivant en conservant les points clés:\n\n{text}' },
            { name: 'Compte-rendu', content: 'Transforme cette transcription en compte-rendu structuré avec des sections et des points clés:\n\n{text}' }
        ],
        dragActive: false,
        selectedFile: null,
        processing: false,
        progress: 0,
        statusText: 'Préparation...',
        result: null,
        resultMeta: null,
        error: null,
        copied: false,
        
        // Speaker naming
        detectedSpeakers: [],
        speakerNames: {},
        originalResult: null,
        
        // Dictation state
        isRecording: false,
        dictationProcessing: false,
        dictationText: '',
        dictationError: null,
        recordingTime: 0,
        mediaRecorder: null,
        audioChunks: [],
        recordingInterval: null,
        chunkInterval: null,
        silenceInterval: null,
        silenceDuration: 0,
        audioLevel: 0,
        pendingTranscription: '',
        dictationOptions: {
            language: '',
            chunkDuration: 5,  // seconds between transcriptions
            silenceThreshold: 0.02,  // audio level threshold for silence (increased)
            silenceTimeout: 10  // seconds of silence before auto-stop
        },
        
        // Options
        options: {
            language: '',
            format: 'json',
            diarize: false,
            minSpeakers: '',
            maxSpeakers: ''
        },
        
        // Save options
        saveOptions: {
            enabled: false,
            folderHandle: null,
            folderName: ''
        },
        
        // Language map for display
        languageNames: {
            'fr': 'Français',
            'en': 'English',
            'es': 'Español',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português',
            'nl': 'Nederlands',
            'pl': 'Polski',
            'ru': 'Русский',
            'ja': '日本語',
            'zh': '中文',
            'ar': 'العربية'
        },
        
        /**
         * Handle file drop event
         */
        handleDrop(event) {
            this.dragActive = false;
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                this.selectFile(files[0]);
            }
        },
        
        /**
         * Handle file input change
         */
        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                this.selectFile(file);
            }
        },
        
        /**
         * Select and validate a file
         */
        selectFile(file) {
            // Validate file type
            const validTypes = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm', 'mp4'];
            const ext = file.name.split('.').pop().toLowerCase();
            
            if (!validTypes.includes(ext)) {
                this.showError(`Format non supporté. Formats acceptés: ${validTypes.join(', ')}`);
                return;
            }
            
            // Validate file size (500MB max)
            const maxSize = 500 * 1024 * 1024;
            if (file.size > maxSize) {
                this.showError('Fichier trop volumineux. Taille max: 500MB');
                return;
            }
            
            this.selectedFile = file;
            this.error = null;
            this.result = null;
            this.resultMeta = null;
        },
        
        /**
         * Remove selected file
         */
        removeFile() {
            this.selectedFile = null;
            // Reset file input
            const input = document.getElementById('fileInput');
            if (input) input.value = '';
        },
        
        /**
         * Start transcription process
         */
        async startTranscription() {
            if (!this.selectedFile || this.processing) return;
            
            this.processing = true;
            this.progress = 0;
            this.error = null;
            this.result = null;
            this.resultMeta = null;
            this.llmResult = '';
            this.statusText = 'Préparation...';
            this.progressInterval = null;
            
            try {
                // Prepare form data
                const formData = new FormData();
                formData.append('file', this.selectedFile);
                formData.append('response_format', this.options.format);
                formData.append('diarize', this.options.diarize);
                
                if (this.options.language) {
                    formData.append('language', this.options.language);
                }
                
                // Speaker constraints
                if (this.options.diarize && this.options.minSpeakers) {
                    formData.append('min_speakers', parseInt(this.options.minSpeakers));
                }
                if (this.options.diarize && this.options.maxSpeakers) {
                    formData.append('max_speakers', parseInt(this.options.maxSpeakers));
                }
                
                // Upload file with real progress tracking
                this.statusText = '📤 Envoi du fichier...';
                const response = await this.uploadWithProgress(formData);
                
                // Process response
                let data;
                const contentType = response.headers.get('content-type');
                
                if (contentType && contentType.includes('application/json')) {
                    data = await response.json();
                    this.resultMeta = {
                        language: data.language,
                        duration: data.duration
                    };
                    this.result = this.options.format === 'json' 
                        ? JSON.stringify(data, null, 2)
                        : data.text;
                    
                    if (this.options.diarize) {
                        this.extractSpeakers(data);
                        this.originalResult = this.result;
                    }
                } else {
                    data = await response.text();
                    this.result = data;
                    
                    if (this.options.diarize) {
                        this.extractSpeakers(null);
                        this.originalResult = this.result;
                    }
                }
                
                this.stopProgressSimulation();
                this.progress = 100;
                this.statusText = '✅ Terminé !';
                
                // Auto-save if enabled
                if (this.saveOptions.enabled && this.result) {
                    const filename = this.generateSaveFilename();
                    await this.saveResultToFile(this.result, filename);
                    this.statusText = '✅ Terminé et sauvegardé !';
                }
                
            } catch (err) {
                console.error('Transcription error:', err);
                this.stopProgressSimulation();
                this.showError(err.message || 'Une erreur est survenue');
            } finally {
                setTimeout(() => {
                    this.processing = false;
                }, 500);
            }
        },
        
        /**
         * Upload file with XMLHttpRequest for real progress tracking
         */
        uploadWithProgress(formData) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                
                // Track upload progress (0-25%)
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const uploadPercent = Math.round((e.loaded / e.total) * 25);
                        this.progress = uploadPercent;
                        this.statusText = `📤 Envoi du fichier... ${Math.round((e.loaded / e.total) * 100)}%`;
                    }
                });
                
                // Upload complete, start processing phase
                xhr.upload.addEventListener('load', () => {
                    this.progress = 25;
                    this.startProcessingProgress();
                });
                
                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        // Create a Response-like object
                        const headers = new Headers();
                        const contentType = xhr.getResponseHeader('content-type');
                        if (contentType) headers.set('content-type', contentType);
                        
                        resolve({
                            ok: true,
                            status: xhr.status,
                            headers: headers,
                            json: () => Promise.resolve(JSON.parse(xhr.responseText)),
                            text: () => Promise.resolve(xhr.responseText)
                        });
                    } else {
                        try {
                            const errorData = JSON.parse(xhr.responseText);
                            reject(new Error(errorData.detail || `Erreur ${xhr.status}`));
                        } catch {
                            reject(new Error(`Erreur ${xhr.status}`));
                        }
                    }
                });
                
                xhr.addEventListener('error', () => reject(new Error('Erreur réseau')));
                xhr.addEventListener('abort', () => reject(new Error('Requête annulée')));
                
                xhr.open('POST', '/v1/audio/transcriptions');
                xhr.send(formData);
            });
        },
        
        /**
         * Start simulated progress during server processing
         */
        startProcessingProgress() {
            const stages = this.options.diarize ? [
                { start: 25, end: 40, text: '🔄 Chargement du modèle Whisper...', duration: 3000 },
                { start: 40, end: 60, text: '🎤 Transcription de l\'audio...', duration: 8000 },
                { start: 60, end: 80, text: '👥 Analyse des locuteurs...', duration: 10000 },
                { start: 80, end: 95, text: '📝 Alignement des segments...', duration: 5000 }
            ] : [
                { start: 25, end: 50, text: '🔄 Chargement du modèle Whisper...', duration: 3000 },
                { start: 50, end: 85, text: '🎤 Transcription de l\'audio...', duration: 10000 },
                { start: 85, end: 95, text: '📝 Formatage du résultat...', duration: 3000 }
            ];
            
            let stageIndex = 0;
            let stageProgress = 0;
            
            this.progressInterval = setInterval(() => {
                if (stageIndex >= stages.length) {
                    // Keep at 95% until response arrives
                    this.progress = 95;
                    this.statusText = '⏳ Finalisation...';
                    return;
                }
                
                const stage = stages[stageIndex];
                const stageRange = stage.end - stage.start;
                const incrementsPerStage = stage.duration / 100;
                
                stageProgress += (stageRange / (stage.duration / 100));
                const currentProgress = stage.start + Math.min(stageProgress, stageRange);
                
                this.progress = Math.round(currentProgress);
                this.statusText = stage.text;
                
                if (currentProgress >= stage.end) {
                    stageIndex++;
                    stageProgress = 0;
                }
            }, 100);
        },
        
        /**
         * Stop progress simulation
         */
        stopProgressSimulation() {
            if (this.progressInterval) {
                clearInterval(this.progressInterval);
                this.progressInterval = null;
            }
        },
        
        /**
         * Show error message
         */
        showError(message) {
            this.error = message;
            this.processing = false;
        },
        
        /**
         * Download transcription result
         */
        downloadResult() {
            if (!this.result) return;
            
            const extensions = {
                'json': 'json',
                'text': 'txt',
                'srt': 'srt',
                'vtt': 'vtt'
            };
            
            const ext = extensions[this.options.format] || 'txt';
            const mimeTypes = {
                'json': 'application/json',
                'txt': 'text/plain',
                'srt': 'text/plain',
                'vtt': 'text/vtt'
            };
            
            const blob = new Blob([this.result], { type: mimeTypes[ext] || 'text/plain' });
            const url = URL.createObjectURL(blob);
            
            const fileName = this.selectedFile 
                ? this.selectedFile.name.replace(/\.[^/.]+$/, '') 
                : 'transcription';
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `${fileName}.${ext}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        },
        
        /**
         * Copy result to clipboard
         */
        async copyToClipboard() {
            if (!this.result) return;
            
            try {
                await navigator.clipboard.writeText(this.result);
                this.copied = true;
                setTimeout(() => {
                    this.copied = false;
                }, 2000);
            } catch (err) {
                console.error('Copy failed:', err);
                this.showError('Impossible de copier dans le presse-papier');
            }
        },
        
        /**
         * Clear result and start fresh
         */
        clearResult() {
            this.result = null;
            this.resultMeta = null;
            this.originalResult = null;
            this.detectedSpeakers = [];
            this.speakerNames = {};
            this.removeFile();
        },
        
        /**
         * Format file size for display
         */
        formatFileSize(bytes) {
            if (!bytes) return '';
            
            const units = ['o', 'Ko', 'Mo', 'Go'];
            let unitIndex = 0;
            let size = bytes;
            
            while (size >= 1024 && unitIndex < units.length - 1) {
                size /= 1024;
                unitIndex++;
            }
            
            return `${size.toFixed(1)} ${units[unitIndex]}`;
        },
        
        /**
         * Format duration for display
         */
        formatDuration(seconds) {
            if (!seconds) return '';
            
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            
            if (mins > 0) {
                return `${mins}m ${secs}s`;
            }
            return `${secs}s`;
        },
        
        /**
         * Get language display name
         */
        getLanguageName(code) {
            if (!code) return 'Inconnu';
            return this.languageNames[code] || code.toUpperCase();
        },
        
        /**
         * Extract unique speakers from result
         */
        extractSpeakers(data) {
            const speakers = new Set();
            
            // Try to extract from JSON segments
            if (data && data.segments) {
                data.segments.forEach(seg => {
                    if (seg.speaker) {
                        speakers.add(seg.speaker);
                    }
                });
            }
            
            // Also extract from text using regex
            if (typeof this.result === 'string') {
                const matches = this.result.match(/SPEAKER_\d+/g) || [];
                matches.forEach(s => speakers.add(s));
            }
            
            // Sort speakers
            this.detectedSpeakers = Array.from(speakers).sort();
            
            // Initialize speaker names
            this.speakerNames = {};
            this.detectedSpeakers.forEach(speaker => {
                this.speakerNames[speaker] = '';
            });
        },
        
        /**
         * Apply speaker names to transcription
         */
        applySpeakerNames() {
            if (!this.originalResult) {
                this.originalResult = this.result;
            }
            
            let newResult = this.originalResult;
            
            // Replace speaker IDs with names
            this.detectedSpeakers.forEach(speaker => {
                const name = this.speakerNames[speaker]?.trim();
                if (name) {
                    // Replace in text
                    const regex = new RegExp(speaker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
                    newResult = newResult.replace(regex, name);
                }
            });
            
            this.result = newResult;
        },
        
        /**
         * Reset speaker names to original
         */
        resetSpeakerNames() {
            if (this.originalResult) {
                this.result = this.originalResult;
            }
            this.detectedSpeakers.forEach(speaker => {
                this.speakerNames[speaker] = '';
            });
        },
        
        /**
         * Check if any speaker has a name
         */
        hasSpeakerNames() {
            return this.detectedSpeakers.some(s => this.speakerNames[s]?.trim());
        },
        
        // ===== DICTATION FUNCTIONS =====
        
        /**
         * Toggle recording on/off
         */
        async toggleRecording() {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                await this.startRecording();
            }
        },
        
        /**
         * Start audio recording with real-time transcription
         */
        async startRecording() {
            try {
                this.dictationError = null;
                this.pendingTranscription = '';
                this._allChunks = []; // Store all chunks for complete file
                this._lastProcessedIndex = 0;
                this.silenceDuration = 0;
                
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
                
                // Store stream reference for cleanup
                this._audioStream = stream;
                
                // Setup audio analysis for silence detection
                this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
                this._analyser = this._audioContext.createAnalyser();
                this._analyser.fftSize = 256;
                const source = this._audioContext.createMediaStreamSource(stream);
                source.connect(this._analyser);
                this._audioDataArray = new Uint8Array(this._analyser.frequencyBinCount);
                
                // Create MediaRecorder
                const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
                    ? 'audio/webm;codecs=opus' 
                    : 'audio/webm';
                    
                this.mediaRecorder = new MediaRecorder(stream, { mimeType });
                this.audioChunks = [];
                
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.audioChunks.push(event.data);
                        this._allChunks.push(event.data);
                    }
                };
                
                this.mediaRecorder.onstop = () => {
                    // Process final complete recording
                    if (this._allChunks.length > 0) {
                        this.processFinalRecording();
                    }
                };
                
                // Start recording - collect data continuously
                this.mediaRecorder.start(1000); // Collect data every second
                this.isRecording = true;
                this.recordingTime = 0;
                
                // Start timer
                this.recordingInterval = setInterval(() => {
                    this.recordingTime++;
                }, 1000);
                
                // Start periodic transcription (only if not in silence)
                this.chunkInterval = setInterval(() => {
                    if (this.isRecording && !this.dictationProcessing && 
                        this.recordingTime >= this.dictationOptions.chunkDuration &&
                        this.silenceDuration < 3) { // Don't transcribe if in silence for 3+ seconds
                        this.processIntermediateRecording();
                    }
                }, this.dictationOptions.chunkDuration * 1000);
                
                // Start silence detection
                this.silenceInterval = setInterval(() => {
                    this.checkSilence();
                }, 500); // Check every 500ms
                
            } catch (err) {
                console.error('Recording error:', err);
                if (err.name === 'NotAllowedError') {
                    this.dictationError = 'Accès au microphone refusé. Veuillez autoriser l\'accès.';
                } else if (err.name === 'NotFoundError') {
                    this.dictationError = 'Aucun microphone trouvé.';
                } else {
                    this.dictationError = `Erreur: ${err.message}`;
                }
            }
        },
        
        /**
         * Stop audio recording
         */
        stopRecording() {
            if (this.mediaRecorder && this.isRecording) {
                this.isRecording = false;
                
                // Stop silence detection
                if (this.silenceInterval) {
                    clearInterval(this.silenceInterval);
                    this.silenceInterval = null;
                }
                
                // Stop chunk processing
                if (this.chunkInterval) {
                    clearInterval(this.chunkInterval);
                    this.chunkInterval = null;
                }
                
                // Stop timer
                if (this.recordingInterval) {
                    clearInterval(this.recordingInterval);
                    this.recordingInterval = null;
                }
                
                // Stop recording (triggers onstop which processes final audio)
                this.mediaRecorder.stop();
                
                // Stop audio stream
                if (this._audioStream) {
                    this._audioStream.getTracks().forEach(track => track.stop());
                    this._audioStream = null;
                }
                
                // Close audio context
                if (this._audioContext) {
                    this._audioContext.close();
                    this._audioContext = null;
                }
                
                this.silenceDuration = 0;
                this.audioLevel = 0;
            }
        },
        
        /**
         * Check for silence and auto-stop if needed
         */
        checkSilence() {
            if (!this._analyser || !this.isRecording) return;
            
            // Get audio level using time domain data (more accurate for voice)
            const dataArray = new Uint8Array(this._analyser.fftSize);
            this._analyser.getByteTimeDomainData(dataArray);
            
            // Calculate RMS (root mean square) for volume
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const sample = (dataArray[i] - 128) / 128; // Normalize to -1 to 1
                sum += sample * sample;
            }
            const rms = Math.sqrt(sum / dataArray.length);
            this.audioLevel = Math.min(rms * 3, 1); // Scale up for visibility
            
            // Check if below threshold
            if (rms < this.dictationOptions.silenceThreshold) {
                this.silenceDuration += 0.5; // Add 500ms
                
                // Auto-stop if silence exceeds timeout and we have some audio recorded
                if (this.silenceDuration >= this.dictationOptions.silenceTimeout && this.recordingTime > 2) {
                    console.log('Auto-stopping due to silence after', this.silenceDuration, 'seconds');
                    this.stopRecording();
                }
            } else {
                // Reset silence counter when sound detected
                this.silenceDuration = 0;
            }
        },
        
        /**
         * Process intermediate recording (all chunks so far)
         */
        async processIntermediateRecording() {
            if (this._allChunks.length === 0) return;
            
            this.dictationProcessing = true;
            
            try {
                // Create complete audio blob from ALL chunks (valid webm file)
                const audioBlob = new Blob(this._allChunks, { type: 'audio/webm' });
                
                // Skip if too small
                if (audioBlob.size < 2000) {
                    this.dictationProcessing = false;
                    return;
                }
                
                // Create form data
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.webm');
                formData.append('response_format', 'json');
                
                if (this.dictationOptions.language) {
                    formData.append('language', this.dictationOptions.language);
                }
                
                // Send to API
                const response = await fetch('/v1/audio/transcriptions', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Erreur ${response.status}`);
                }
                
                const data = await response.json();
                
                // Replace with full transcription (not append, since we send all audio)
                if (data.text && data.text.trim()) {
                    this.dictationText = data.text.trim();
                }
                
            } catch (err) {
                console.error('Intermediate processing error:', err);
                // Don't show errors for intermediate processing
            } finally {
                this.dictationProcessing = false;
            }
        },
        
        /**
         * Process final complete recording
         */
        async processFinalRecording() {
            if (this._allChunks.length === 0) return;
            
            this.dictationProcessing = true;
            
            try {
                // Create complete audio blob
                const audioBlob = new Blob(this._allChunks, { type: 'audio/webm' });
                
                if (audioBlob.size < 1000) {
                    this.dictationProcessing = false;
                    return;
                }
                
                // Create form data
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.webm');
                formData.append('response_format', 'json');
                
                if (this.dictationOptions.language) {
                    formData.append('language', this.dictationOptions.language);
                }
                
                // Send to API
                const response = await fetch('/v1/audio/transcriptions', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Erreur ${response.status}`);
                }
                
                const data = await response.json();
                
                // Set final transcription
                if (data.text && data.text.trim()) {
                    this.dictationText = data.text.trim();
                }
                
            } catch (err) {
                console.error('Final processing error:', err);
                this.dictationError = err.message || 'Erreur lors du traitement final';
            } finally {
                this.dictationProcessing = false;
                this._allChunks = [];
            }
        },
        
        /**
         * Format recording time as MM:SS
         */
        formatRecordingTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        },
        
        /**
         * Copy dictation text to clipboard
         */
        async copyDictation() {
            if (!this.dictationText) return;
            
            try {
                await navigator.clipboard.writeText(this.dictationText);
                // Could add a visual feedback here
            } catch (err) {
                console.error('Copy failed:', err);
                this.dictationError = 'Impossible de copier dans le presse-papier';
            }
        },
        
        /**
         * Clear dictation text
         */
        clearDictation() {
            this.dictationText = '';
            this.dictationError = null;
        },
        
        // ===== OLLAMA FUNCTIONS =====
        
        /**
         * Initialize - load saved settings
         */
        init() {
            this.loadSettings();
            // Auto-test connection if URL is set
            if (this.ollamaSettings.url) {
                this.testOllamaConnection();
            }
        },
        
        /**
         * Test Ollama connection and fetch models
         */
        async testOllamaConnection() {
            this.ollamaLoading = true;
            this.ollamaStatus = '';
            this.ollamaConnected = false;
            
            try {
                // Fetch models from Ollama API
                const response = await fetch(`${this.ollamaSettings.url}/api/tags`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                
                // Extract model names
                this.ollamaModels = (data.models || []).map(m => m.name);
                
                if (this.ollamaModels.length === 0) {
                    this.ollamaStatus = 'Connecté mais aucun modèle trouvé';
                    this.ollamaConnected = true;
                } else {
                    this.ollamaStatus = `Connecté - ${this.ollamaModels.length} modèle(s)`;
                    this.ollamaConnected = true;
                    
                    // Auto-select first model if none selected
                    if (!this.ollamaSettings.model && this.ollamaModels.length > 0) {
                        this.ollamaSettings.model = this.ollamaModels[0];
                    }
                }
                
            } catch (err) {
                console.error('Ollama connection error:', err);
                this.ollamaStatus = `Erreur: ${err.message}`;
                this.ollamaConnected = false;
                this.ollamaModels = [];
            } finally {
                this.ollamaLoading = false;
            }
        },
        
        /**
         * Apply LLM prompt to transcription
         */
        async applyLLMPrompt(promptIndex) {
            if (!this.ollamaConnected || !this.ollamaSettings.model) {
                this.error = 'Configurez Ollama dans l\'onglet Paramètres';
                return;
            }
            
            const prompt = this.customPrompts[promptIndex];
            if (!prompt) return;
            
            // Get the text to process
            const textToProcess = this.getPlainTextResult();
            if (!textToProcess) {
                this.error = 'Aucun texte à traiter';
                return;
            }
            
            this.llmProcessing = true;
            this.llmResult = '';
            this.llmLastPrompt = prompt.name || `Prompt ${promptIndex + 1}`;
            
            try {
                // Replace {text} placeholder with actual transcription
                const fullPrompt = prompt.content.replace('{text}', textToProcess);
                
                // Call Ollama API
                const response = await fetch(`${this.ollamaSettings.url}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.ollamaSettings.model,
                        prompt: fullPrompt,
                        stream: false
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                
                // Store LLM result separately
                if (data.response) {
                    this.llmResult = data.response;
                }
                
            } catch (err) {
                console.error('LLM processing error:', err);
                this.error = `Erreur LLM: ${err.message}`;
            } finally {
                this.llmProcessing = false;
            }
        },
        
        /**
         * Get plain text from result (remove JSON formatting if needed)
         */
        getPlainTextResult() {
            if (!this.result) return '';
            
            try {
                // Try to parse as JSON and extract text
                const parsed = JSON.parse(this.result);
                return parsed.text || this.result;
            } catch {
                // Not JSON, return as-is
                return this.result;
            }
        },
        
        /**
         * Format LLM result for display (convert markdown-like to HTML)
         */
        formatLLMResultForDisplay() {
            if (!this.llmResult) return '';
            
            let html = this.llmResult
                // Escape HTML first
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                // Bold text **text** or __text__
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/__(.+?)__/g, '<strong>$1</strong>')
                // Italic text *text* or _text_
                .replace(/\*([^*]+)\*/g, '<em>$1</em>')
                .replace(/_([^_]+)_/g, '<em>$1</em>')
                // Headers
                .replace(/^### (.+)$/gm, '<h4>$1</h4>')
                .replace(/^## (.+)$/gm, '<h3>$1</h3>')
                .replace(/^# (.+)$/gm, '<h2>$1</h2>')
                // Bullet points
                .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
                // Numbered lists
                .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
                // Paragraphs (double newlines)
                .replace(/\n\n/g, '</p><p>')
                // Single newlines to <br>
                .replace(/\n/g, '<br>');
            
            // Wrap in paragraph
            html = '<p>' + html + '</p>';
            
            // Wrap consecutive <li> in <ul>
            html = html.replace(/(<li>.*?<\/li>)+/gs, '<ul>$&</ul>');
            
            // Clean up empty paragraphs
            html = html.replace(/<p><\/p>/g, '');
            html = html.replace(/<p><br><\/p>/g, '');
            
            return html;
        },
        
        /**
         * Copy LLM result formatted for Word/Outlook
         */
        async copyLLMResultFormatted() {
            if (!this.llmResult) return;
            
            try {
                // Create formatted HTML for clipboard
                const htmlContent = this.createWordFormattedHTML();
                const plainText = this.llmResult;
                
                // Use Clipboard API with both formats
                const clipboardItem = new ClipboardItem({
                    'text/html': new Blob([htmlContent], { type: 'text/html' }),
                    'text/plain': new Blob([plainText], { type: 'text/plain' })
                });
                
                await navigator.clipboard.write([clipboardItem]);
                
                this.llmCopied = true;
                setTimeout(() => { this.llmCopied = false; }, 2000);
            } catch (err) {
                console.error('Failed to copy formatted:', err);
                // Fallback to plain text
                try {
                    await navigator.clipboard.writeText(this.llmResult);
                    this.llmCopied = true;
                    setTimeout(() => { this.llmCopied = false; }, 2000);
                } catch (e) {
                    console.error('Fallback copy failed:', e);
                }
            }
        },
        
        /**
         * Create Word/Outlook compatible HTML
         */
        createWordFormattedHTML() {
            const text = this.llmResult;
            
            // Start with proper HTML structure for Word
            let html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }
p { margin: 0 0 10pt 0; }
h1 { font-size: 16pt; font-weight: bold; margin: 12pt 0 6pt 0; color: #1a1a1a; }
h2 { font-size: 14pt; font-weight: bold; margin: 10pt 0 5pt 0; color: #1a1a1a; }
h3 { font-size: 12pt; font-weight: bold; margin: 8pt 0 4pt 0; color: #1a1a1a; }
h4 { font-size: 11pt; font-weight: bold; margin: 6pt 0 3pt 0; color: #1a1a1a; }
ul, ol { margin: 6pt 0 6pt 20pt; padding: 0; }
li { margin: 3pt 0; }
strong { font-weight: bold; }
em { font-style: italic; }
</style>
</head>
<body>`;
            
            // Process content
            let content = text
                // Escape HTML entities
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                // Bold
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/__(.+?)__/g, '<strong>$1</strong>')
                // Italic
                .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
                .replace(/_([^_\n]+)_/g, '<em>$1</em>')
                // Headers
                .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
                .replace(/^### (.+)$/gm, '<h3>$1</h3>')
                .replace(/^## (.+)$/gm, '<h2>$1</h2>')
                .replace(/^# (.+)$/gm, '<h1>$1</h1>');
            
            // Process paragraphs and lists
            const lines = content.split('\n');
            let result = [];
            let inList = false;
            let listType = null;
            
            for (let i = 0; i < lines.length; i++) {
                let line = lines[i].trim();
                
                // Skip empty lines but close list if needed
                if (!line) {
                    if (inList) {
                        result.push(listType === 'ul' ? '</ul>' : '</ol>');
                        inList = false;
                        listType = null;
                    }
                    continue;
                }
                
                // Check for bullet points
                const bulletMatch = line.match(/^[\-\*•] (.+)$/);
                const numberMatch = line.match(/^\d+[\.\)] (.+)$/);
                
                if (bulletMatch) {
                    if (!inList || listType !== 'ul') {
                        if (inList) result.push(listType === 'ul' ? '</ul>' : '</ol>');
                        result.push('<ul>');
                        inList = true;
                        listType = 'ul';
                    }
                    result.push(`<li>${bulletMatch[1]}</li>`);
                } else if (numberMatch) {
                    if (!inList || listType !== 'ol') {
                        if (inList) result.push(listType === 'ul' ? '</ul>' : '</ol>');
                        result.push('<ol>');
                        inList = true;
                        listType = 'ol';
                    }
                    result.push(`<li>${numberMatch[1]}</li>`);
                } else {
                    // Close list if needed
                    if (inList) {
                        result.push(listType === 'ul' ? '</ul>' : '</ol>');
                        inList = false;
                        listType = null;
                    }
                    
                    // Check if it's already a header tag
                    if (line.startsWith('<h')) {
                        result.push(line);
                    } else {
                        result.push(`<p>${line}</p>`);
                    }
                }
            }
            
            // Close any remaining list
            if (inList) {
                result.push(listType === 'ul' ? '</ul>' : '</ol>');
            }
            
            html += result.join('\n');
            html += '</body></html>';
            
            return html;
        },
        
        /**
         * Clear LLM result
         */
        clearLLMResult() {
            this.llmResult = '';
            this.llmLastPrompt = '';
            this.llmCopied = false;
        },
        
        /**
         * Choose save folder using File System Access API
         */
        async chooseSaveFolder() {
            try {
                // Check if File System Access API is supported
                if ('showDirectoryPicker' in window) {
                    const handle = await window.showDirectoryPicker({
                        mode: 'readwrite'
                    });
                    this.saveOptions.folderHandle = handle;
                    this.saveOptions.folderName = handle.name;
                    
                    // Save to localStorage for persistence
                    localStorage.setItem('mywhisper_save_folder_name', handle.name);
                } else {
                    alert('Votre navigateur ne supporte pas la sélection de dossier.\nLes fichiers seront téléchargés dans votre dossier Téléchargements par défaut.');
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    console.error('Folder selection error:', err);
                }
            }
        },
        
        /**
         * Save result to file
         */
        async saveResultToFile(content, filename) {
            if (!this.saveOptions.enabled) return;
            
            try {
                if (this.saveOptions.folderHandle) {
                    // Use File System Access API to save directly to folder
                    await this.saveToFolder(content, filename);
                } else {
                    // Fallback to standard download
                    this.downloadFile(content, filename);
                }
            } catch (err) {
                console.error('Save error:', err);
                // Fallback to standard download on error
                this.downloadFile(content, filename);
            }
        },
        
        /**
         * Save file directly to selected folder
         */
        async saveToFolder(content, filename) {
            try {
                const fileHandle = await this.saveOptions.folderHandle.getFileHandle(filename, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(content);
                await writable.close();
                console.log(`File saved: ${filename}`);
            } catch (err) {
                console.error('Error saving to folder:', err);
                // Permission might have been revoked, reset folder handle
                if (err.name === 'NotAllowedError') {
                    this.saveOptions.folderHandle = null;
                    this.saveOptions.folderName = '';
                    alert('Permission refusée. Veuillez resélectionner le dossier.');
                }
                throw err;
            }
        },
        
        /**
         * Standard file download
         */
        downloadFile(content, filename) {
            const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        },
        
        /**
         * Generate filename for save
         */
        generateSaveFilename() {
            const baseName = this.selectedFile ? 
                this.selectedFile.name.replace(/\.[^/.]+$/, '') : 
                'transcription';
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
            const extension = this.getFileExtension();
            return `${baseName}_${timestamp}${extension}`;
        },
        
        /**
         * Get file extension based on format
         */
        getFileExtension() {
            switch (this.options.format) {
                case 'srt': return '.srt';
                case 'vtt': return '.vtt';
                case 'json': return '.json';
                default: return '.txt';
            }
        },
        
        /**
         * Add a new custom prompt
         */
        addPrompt() {
            this.customPrompts.push({
                name: '',
                content: ''
            });
        },
        
        /**
         * Remove a custom prompt
         */
        removePrompt(index) {
            this.customPrompts.splice(index, 1);
        },
        
        /**
         * Save settings to localStorage
         */
        saveSettings() {
            const settings = {
                ollamaSettings: this.ollamaSettings,
                customPrompts: this.customPrompts
            };
            localStorage.setItem('mywhisper_settings', JSON.stringify(settings));
            alert('Paramètres sauvegardés !');
        },
        
        /**
         * Load settings from localStorage
         */
        loadSettings() {
            try {
                const saved = localStorage.getItem('mywhisper_settings');
                if (saved) {
                    const settings = JSON.parse(saved);
                    if (settings.ollamaSettings) {
                        this.ollamaSettings = { ...this.ollamaSettings, ...settings.ollamaSettings };
                    }
                    if (settings.customPrompts && settings.customPrompts.length > 0) {
                        this.customPrompts = settings.customPrompts;
                    }
                }
            } catch (err) {
                console.error('Failed to load settings:', err);
            }
        }
    };
}
