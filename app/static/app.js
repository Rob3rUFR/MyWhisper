/**
 * MyWhisper - Speech to Text Application
 * Alpine.js application for handling audio transcription
 */

function sttApp() {
    return {
        // State
        activeTab: 'file',
        
        // Ollama settings (configured via .env on server)
        ollamaSettings: {
            model: ''
        },
        ollamaModels: [],
        ollamaConnected: false,
        ollamaConfigured: false,
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
        speakerSamples: {},  // Speaker audio samples info
        sessionId: null,     // Session ID for audio sample API
        historyId: null,     // History record ID for speaker naming sync
        clientId: null,      // Client ID for result recovery after disconnect
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
        
        // Processing status (global lock)
        serverProcessing: {
            is_processing: false,
            current_file: null,
            processing_type: null,
            elapsed_seconds: null,
            cancel_requested: false,
            current_step: 'idle',
            progress_percent: 0,
            total_chunks: 0,
            current_chunk: 0,
            audio_duration: 0
        },
        statusCheckInterval: null,
        cancelRequested: false,
        useServerProgress: false,  // Use server-reported progress when available
        
        // Options
        options: {
            language: '',
            format: 'text',
            diarize: false,
            minSpeakers: '',
            maxSpeakers: ''
        },
        
        // History state
        historyItems: [],
        historyTotal: 0,
        historyOffset: 0,
        historyLimit: 20,
        historyLoading: false,
        historyViewItem: null,
        
        // Settings state
        retentionDays: 90,
        retentionLoading: false,
        
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
            
            // Check if server is available
            await this.checkServerStatus();
            if (this.serverProcessing.is_processing) {
                this.showError(this.getBlockingMessage());
                return;
            }
            
            this.processing = true;
            this.progress = 0;
            this.error = null;
            this.result = null;
            this.resultMeta = null;
            this.llmResult = '';
            this.statusText = 'Préparation...';
            this.progressInterval = null;
            this.cancelRequested = false;
            this.useServerProgress = false;
            
            // Generate unique client ID for result recovery
            this.clientId = crypto.randomUUID();
            console.log('Generated client ID:', this.clientId);
            
            try {
                // Prepare form data
                const formData = new FormData();
                formData.append('file', this.selectedFile);
                formData.append('response_format', this.options.format);
                formData.append('diarize', this.options.diarize);
                formData.append('client_id', this.clientId);
                
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
                        // Use speaker metadata from SSE response for non-JSON formats
                        const speakerMeta = response._speakerMeta || {};
                        this.extractSpeakers(null, speakerMeta);
                        this.originalResult = this.result;
                    }
                }
                
                this.stopProgressSimulation();
                this.progress = 100;
                this.statusText = '✅ Terminé !';
                
                // Save state after successful transcription
                this.saveState();
                
            } catch (err) {
                console.error('Transcription error:', err);
                this.stopProgressSimulation();
                
                // Don't show error if cancelled
                if (err.message !== 'Traitement annulé') {
                    this.showError(err.message || 'Une erreur est survenue');
                } else {
                    this.statusText = '⏹️ Traitement annulé';
                }
            } finally {
                setTimeout(() => {
                    this.processing = false;
                    this.cancelRequested = false;
                    // Save final state
                    this.saveState();
                }, 500);
            }
        },
        
        /**
         * Upload file with SSE streaming to prevent 504 timeout errors.
         * Uses the streaming endpoint that sends progress updates.
         */
        uploadWithProgress(formData) {
            return new Promise((resolve, reject) => {
                // First upload the file with XHR to track upload progress
                const xhr = new XMLHttpRequest();
                
                // Track upload progress (0-25%)
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const uploadPercent = Math.round((e.loaded / e.total) * 25);
                        this.progress = uploadPercent;
                        this.statusText = `📤 Envoi du fichier... ${Math.round((e.loaded / e.total) * 100)}%`;
                    }
                });
                
                // Handle SSE response for processing progress
                xhr.onreadystatechange = () => {
                    if (xhr.readyState === XMLHttpRequest.HEADERS_RECEIVED) {
                        const contentType = xhr.getResponseHeader('content-type');
                        
                        if (contentType && contentType.includes('text/event-stream')) {
                            // SSE streaming response - process events as they arrive
                            this.progress = 25;
                            this.statusText = '🔄 Traitement en cours...';
                        }
                    }
                    
                    if (xhr.readyState === XMLHttpRequest.LOADING) {
                        const contentType = xhr.getResponseHeader('content-type');
                        if (contentType && contentType.includes('text/event-stream')) {
                            // Parse SSE events from partial response
                            this.processSSEChunk(xhr.responseText);
                        }
                    }
                };
                
                xhr.addEventListener('load', () => {
                    const contentType = xhr.getResponseHeader('content-type');
                    
                    if (contentType && contentType.includes('text/event-stream')) {
                        // Process final SSE result
                        const result = this.extractSSEResult(xhr.responseText);
                        
                        if (result.error) {
                            reject(new Error(result.error));
                        } else if (result.cancelled) {
                            reject(new Error('Traitement annulé'));
                        } else if (result.content !== undefined) {
                            // Create Response-like object
                            const headers = new Headers();
                            const isJson = result.format === 'json';
                            headers.set('content-type', isJson ? 'application/json' : 'text/plain');
                            
                            // Build data object - only parse as JSON if format is json
                            let jsonData = null;
                            if (isJson) {
                                try {
                                    jsonData = typeof result.content === 'string' ? JSON.parse(result.content) : result.content;
                                    // Add speaker samples to JSON data
                                    if (jsonData && typeof jsonData === 'object' && !Array.isArray(jsonData)) {
                                        if (result.session_id) jsonData.session_id = result.session_id;
                                        if (result.speaker_samples) jsonData.speaker_samples = result.speaker_samples;
                                    }
                                } catch (e) {
                                    console.error('Failed to parse JSON content:', e);
                                    jsonData = result.content;
                                }
                            } else {
                                // For text/srt/vtt formats, content is plain text
                                jsonData = result.content;
                            }
                            
                            resolve({
                                ok: true,
                                status: 200,
                                headers: headers,
                                json: () => isJson ? Promise.resolve(jsonData) : Promise.reject(new Error('Not JSON')),
                                text: () => Promise.resolve(
                                    typeof result.content === 'string' ? result.content : JSON.stringify(result.content)
                                ),
                                // Store metadata separately for non-JSON formats
                                _speakerMeta: {
                                    session_id: result.session_id,
                                    speaker_samples: result.speaker_samples,
                                    history_id: result.history_id
                                }
                            });
                        } else {
                            reject(new Error('Réponse invalide du serveur'));
                        }
                    } else if (xhr.status >= 200 && xhr.status < 300) {
                        // Standard JSON response (fallback)
                        const headers = new Headers();
                        if (contentType) headers.set('content-type', contentType);
                        
                        resolve({
                            ok: true,
                            status: xhr.status,
                            headers: headers,
                            json: () => Promise.resolve(JSON.parse(xhr.responseText)),
                            text: () => Promise.resolve(xhr.responseText)
                        });
                    } else if (xhr.status === 499) {
                        reject(new Error('Traitement annulé'));
                    } else if (xhr.status === 503) {
                        // Server busy - try to parse error from SSE or JSON
                        try {
                            if (contentType && contentType.includes('text/event-stream')) {
                                const result = this.extractSSEResult(xhr.responseText);
                                reject(new Error(result.error || 'Le serveur est occupé'));
                            } else {
                                const errorData = JSON.parse(xhr.responseText);
                                reject(new Error(errorData.detail || 'Le serveur est occupé'));
                            }
                        } catch {
                            reject(new Error('Le serveur est occupé par un autre traitement'));
                        }
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
                
                // Use streaming endpoint to prevent 504 timeout
                xhr.open('POST', '/v1/audio/transcriptions/stream');
                xhr.send(formData);
            });
        },
        
        /**
         * Process SSE chunk and update progress
         */
        processSSEChunk(text) {
            const lines = text.split('\n');
            let currentEvent = null;
            let currentData = '';
            
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.substring(7).trim();
                } else if (line.startsWith('data: ')) {
                    currentData = line.substring(6);
                    
                    if (currentEvent === 'progress' && currentData) {
                        try {
                            const progress = JSON.parse(currentData);
                            this.updateProgressFromSSE(progress);
                        } catch (e) {
                            console.debug('Failed to parse progress:', e);
                        }
                    }
                    
                    currentEvent = null;
                    currentData = '';
                }
            }
        },
        
        /**
         * Update progress from SSE event
         */
        updateProgressFromSSE(progress) {
            if (progress.percent !== undefined) {
                this.progress = progress.percent;
                this.useServerProgress = true;
            }
            
            // Don't update status text if cancellation was requested
            if (this.cancelRequested) {
                return;
            }
            
            const step = progress.step;
            switch (step) {
                case 'transcribing':
                    if (progress.audio_duration > 0) {
                        const mins = Math.floor(progress.audio_duration / 60);
                        const secs = Math.round(progress.audio_duration % 60);
                        this.statusText = mins > 0 
                            ? `🎤 Transcription en cours... (${mins}m${secs}s d'audio)`
                            : `🎤 Transcription en cours... (${secs}s d'audio)`;
                    } else {
                        this.statusText = '🎤 Transcription en cours...';
                    }
                    break;
                case 'diarizing_loading':
                    this.statusText = '👥 Chargement du modèle de diarisation...';
                    break;
                case 'diarizing_processing':
                    const subProgress = progress.percent >= 40 ? Math.round(((progress.percent - 40) / 45) * 100) : 0;
                    this.statusText = `👥 Analyse des voix... ${Math.min(subProgress, 100)}%`;
                    break;
                case 'diarizing_chunk':
                    if (progress.total_chunks > 0) {
                        this.statusText = `👥 Diarisation: segment ${progress.current_chunk}/${progress.total_chunks}`;
                    } else {
                        this.statusText = '👥 Analyse des locuteurs...';
                    }
                    break;
                case 'diarizing_merging':
                    this.statusText = '👥 Harmonisation des locuteurs...';
                    break;
                case 'merging':
                    this.statusText = '🔗 Fusion transcription et locuteurs...';
                    break;
                case 'finalizing':
                    this.statusText = '📝 Finalisation...';
                    break;
                case 'complete':
                    this.statusText = '✅ Terminé !';
                    break;
            }
        },
        
        /**
         * Extract final result from SSE response
         */
        extractSSEResult(text) {
            const lines = text.split('\n');
            let result = {};
            let currentEvent = null;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                
                if (line.startsWith('event: ')) {
                    currentEvent = line.substring(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const data = line.substring(6);
                    
                    try {
                        const parsed = JSON.parse(data);
                        
                        if (currentEvent === 'result') {
                            result.content = parsed.content;
                            result.format = parsed.format;
                            // Extract speaker samples metadata
                            if (parsed.session_id) {
                                result.session_id = parsed.session_id;
                            }
                            if (parsed.speaker_samples) {
                                result.speaker_samples = parsed.speaker_samples;
                            }
                            // Extract history ID for speaker naming sync
                            if (parsed.history_id) {
                                result.history_id = parsed.history_id;
                            }
                        } else if (currentEvent === 'error') {
                            result.error = parsed.detail || 'Erreur inconnue';
                        } else if (currentEvent === 'cancelled') {
                            result.cancelled = true;
                        }
                    } catch (e) {
                        console.debug('Failed to parse SSE data:', e);
                    }
                    
                    currentEvent = null;
                }
            }
            
            return result;
        },
        
        /**
         * Start simulated progress during server processing
         * Uses asymptotic progression that slows down as it approaches the limit
         * This provides a more realistic feel for variable-length processing
         * 
         * Progress distribution (with diarization):
         * - 0-5%: Upload complete, starting
         * - 5-30%: Transcription
         * - 30-35%: Loading diarization model
         * - 35-85%: Diarization processing
         * - 85-90%: Merging
         * - 90-100%: Finalizing
         */
        startProcessingProgress() {
            // Estimate processing time based on file size (rough heuristic)
            const fileSizeMB = this.selectedFile ? this.selectedFile.size / (1024 * 1024) : 10;
            // With diarization, processing takes much longer
            const baseFactor = this.options.diarize ? 8 : 3;
            const estimatedSeconds = Math.max(15, fileSizeMB * baseFactor);
            
            // Messages to display during processing (fallback when server doesn't report)
            const transcriptionMessages = [
                '🎤 Analyse de l\'audio...',
                '🎤 Transcription en cours...',
                '🎤 Détection de la parole...',
                '🎤 Conversion audio → texte...',
                '🎤 Traitement des segments...'
            ];
            
            const diarizationLoadingMessages = [
                '👥 Chargement du modèle de diarisation...',
                '👥 Préparation de l\'analyse des voix...'
            ];
            
            const diarizationMessages = [
                '👥 Analyse des voix...',
                '👥 Identification des locuteurs...',
                '👥 Séparation des intervenants...',
                '👥 Détection des changements de locuteur...',
                '👥 Attribution des segments...'
            ];
            
            const mergingMessages = [
                '🔗 Fusion des résultats...',
                '🔗 Association texte et locuteurs...'
            ];
            
            const finalizationMessages = [
                '📝 Formatage du résultat...',
                '📝 Finalisation...',
                '⏳ Presque terminé...'
            ];
            
            let elapsedTime = 0;
            let lastMessageChange = 0;
            let messageIndex = 0;
            // Max progress depends on whether we're using diarization
            const maxProgress = this.options.diarize ? 88 : 90;
            
            // Asymptotic function with adjusted speed for diarization
            const k = 2.5 / estimatedSeconds;
            
            this.progressInterval = setInterval(() => {
                // If server is reporting progress, use that instead of simulation
                if (this.useServerProgress) {
                    return; // Server progress is being used, skip simulation
                }
                
                elapsedTime += 0.2; // 200ms intervals
                
                // Calculate asymptotic progress (25% to maxProgress)
                const asymptotic = 1 - Math.exp(-k * elapsedTime);
                const progressRange = maxProgress - 25;
                let currentProgress = 25 + (progressRange * asymptotic);
                
                // Add small random variations for realism (smaller jitter)
                const jitter = (Math.random() - 0.5) * 0.3;
                currentProgress = Math.min(maxProgress, currentProgress + jitter);
                
                this.progress = Math.round(currentProgress);
                
                // Change status message periodically (more frequent updates)
                const shouldChangeMessage = elapsedTime - lastMessageChange > 3 + Math.random() * 2;
                
                if (shouldChangeMessage) {
                    lastMessageChange = elapsedTime;
                    
                    // Determine which phase we're in based on progress (aligned with server steps)
                    if (this.progress < 30) {
                        // Transcription phase (5-30%)
                        this.statusText = transcriptionMessages[messageIndex % transcriptionMessages.length];
                    } else if (this.options.diarize && this.progress < 38) {
                        // Loading diarization model (30-38%)
                        this.statusText = diarizationLoadingMessages[messageIndex % diarizationLoadingMessages.length];
                    } else if (this.options.diarize && this.progress < 85) {
                        // Diarization processing (38-85%)
                        const subProgress = Math.round(((this.progress - 38) / 47) * 100);
                        this.statusText = `${diarizationMessages[messageIndex % diarizationMessages.length]} ${subProgress}%`;
                    } else if (this.options.diarize && this.progress < 90) {
                        // Merging (85-90%)
                        this.statusText = mergingMessages[messageIndex % mergingMessages.length];
                    } else if (this.progress >= 88) {
                        // Finalizing
                        this.statusText = finalizationMessages[Math.min(messageIndex % finalizationMessages.length, finalizationMessages.length - 1)];
                    } else {
                        // Default to transcription messages for non-diarization
                        this.statusText = transcriptionMessages[messageIndex % transcriptionMessages.length];
                    }
                    messageIndex++;
                }
                
                // Initial message
                if (elapsedTime < 0.5) {
                    this.statusText = '🔄 Initialisation du modèle...';
                }
                
            }, 200);
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
            // Cleanup speaker samples cache
            this.cleanupSpeakerSamples();
            
            this.result = null;
            this.resultMeta = null;
            this.originalResult = null;
            this.detectedSpeakers = [];
            this.speakerNames = {};
            this.speakerSamples = {};
            this.sessionId = null;
            this.historyId = null;
            this.llmResult = '';
            this.llmLastPrompt = '';
            this._restoredFileInfo = null;
            this.removeFile();
            
            // Clear saved state
            this.clearSavedState();
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
         * Format processing duration for display (more detailed for history)
         */
        formatProcessingDuration(seconds) {
            if (!seconds) return '';
            
            if (seconds < 60) {
                return `${Math.round(seconds)}s`;
            }
            
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            
            if (mins < 60) {
                return `${mins}m ${secs}s`;
            }
            
            const hours = Math.floor(mins / 60);
            const remainMins = mins % 60;
            return `${hours}h ${remainMins}m`;
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
         * @param {Object} data - JSON response data (can be null for non-JSON formats)
         * @param {Object} speakerMeta - Optional speaker metadata from SSE response
         */
        extractSpeakers(data, speakerMeta = {}) {
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
            
            // Extract speaker samples info from response data or speakerMeta
            const samples = data?.speaker_samples || speakerMeta.speaker_samples;
            const sessionId = data?.session_id || speakerMeta.session_id;
            const historyId = data?.history_id || speakerMeta.history_id;
            
            if (samples) {
                this.speakerSamples = samples;
                this.sessionId = sessionId;
                console.log('Speaker samples available:', Object.keys(this.speakerSamples));
            } else {
                this.speakerSamples = {};
                this.sessionId = null;
            }
            
            // Store history ID for speaker naming sync
            this.historyId = historyId || null;
            if (historyId) {
                console.log('History ID for speaker naming:', historyId);
            }
        },
        
        /**
         * Get the audio sample URL for a speaker
         */
        getSpeakerSampleUrl(speaker) {
            if (!this.sessionId || !this.speakerSamples[speaker]) {
                return null;
            }
            return `/speaker-sample/${this.sessionId}/${speaker}`;
        },
        
        /**
         * Check if a speaker has an audio sample available
         */
        hasSpeakerSample(speaker) {
            return this.sessionId && this.speakerSamples[speaker];
        },
        
        /**
         * Get the sample text preview for a speaker
         */
        getSpeakerSampleText(speaker) {
            const sample = this.speakerSamples[speaker];
            if (!sample || !sample.text) return '';
            
            // Truncate if too long
            const text = sample.text;
            if (text.length > 80) {
                return text.substring(0, 77) + '...';
            }
            return text;
        },
        
        /**
         * Get the sample duration for a speaker
         */
        getSpeakerSampleDuration(speaker) {
            const sample = this.speakerSamples[speaker];
            if (!sample) return '';
            return `${sample.duration}s`;
        },
        
        /**
         * Play a speaker's audio sample
         */
        playSpeakerSample(speaker) {
            const url = this.getSpeakerSampleUrl(speaker);
            if (!url) return;
            
            // Stop any currently playing audio
            const existingAudio = document.querySelector('audio.speaker-audio-player');
            if (existingAudio) {
                existingAudio.pause();
                existingAudio.remove();
            }
            
            // Create and play new audio
            const audio = new Audio(url);
            audio.className = 'speaker-audio-player';
            audio.style.display = 'none';
            document.body.appendChild(audio);
            
            audio.play().catch(err => {
                console.error('Failed to play speaker sample:', err);
            });
            
            // Remove when finished
            audio.addEventListener('ended', () => {
                audio.remove();
            });
        },
        
        /**
         * Cleanup speaker samples cache when done
         */
        async cleanupSpeakerSamples() {
            if (!this.sessionId) return;
            
            try {
                await fetch(`/speaker-samples/${this.sessionId}`, {
                    method: 'DELETE'
                });
                console.log('Speaker samples cache cleaned up');
            } catch (err) {
                console.debug('Failed to cleanup speaker samples:', err);
            }
            
            this.sessionId = null;
            this.speakerSamples = {};
        },
        
        /**
         * Apply speaker names to transcription
         */
        async applySpeakerNames() {
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
            
            // Update history if we have a history ID
            if (this.historyId && this.hasSpeakerNames()) {
                try {
                    const response = await fetch(`/history/${this.historyId}/speakers`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ speaker_names: this.speakerNames })
                    });
                    
                    if (response.ok) {
                        console.log('Speaker names saved to history');
                    } else {
                        console.warn('Failed to save speaker names to history');
                    }
                } catch (err) {
                    console.warn('Failed to sync speaker names with history:', err);
                }
            }
            
            // Save state after applying names
            this.saveState();
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
            // Check if server is available before starting recording
            await this.checkServerStatus();
            if (this.serverProcessing.is_processing) {
                this.dictationError = this.getBlockingMessage();
                return;
            }
            
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
            
            // Check server status first
            await this.checkServerStatus();
            if (this.serverProcessing.is_processing && this.serverProcessing.processing_type !== 'dictation') {
                // Another file is being processed, skip this chunk
                console.log('Skipping intermediate transcription: server busy with file');
                return;
            }
            
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
                formData.append('file', audioBlob, 'dictation.webm');
                formData.append('response_format', 'json');
                formData.append('processing_type', 'dictation');  // Don't save to history
                
                if (this.dictationOptions.language) {
                    formData.append('language', this.dictationOptions.language);
                }
                
                // Send to API
                const response = await fetch('/v1/audio/transcriptions', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    // If server is busy (503), just skip this chunk
                    if (response.status === 503) {
                        console.log('Server busy, skipping intermediate transcription');
                        return;
                    }
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
                formData.append('file', audioBlob, 'dictation_final.webm');
                formData.append('response_format', 'json');
                formData.append('processing_type', 'dictation');  // Don't save to history
                
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
                    // Show specific message for busy server
                    if (response.status === 503) {
                        throw new Error('Le serveur est occupé par un autre traitement. Veuillez réessayer.');
                    }
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
        
        // ===== STATE PERSISTENCE =====
        
        /**
         * Save current state to sessionStorage for persistence across page refresh
         */
        saveState() {
            const state = {
                // Results
                result: this.result,
                resultMeta: this.resultMeta,
                originalResult: this.originalResult,
                
                // Speaker data
                detectedSpeakers: this.detectedSpeakers,
                speakerNames: this.speakerNames,
                speakerSamples: this.speakerSamples,
                sessionId: this.sessionId,
                historyId: this.historyId,
                clientId: this.clientId,
                
                // Options
                options: this.options,
                
                // File info (not the file itself)
                selectedFileName: this.selectedFile?.name || null,
                selectedFileSize: this.selectedFile?.size || null,
                
                // Progress state (if processing)
                processing: this.processing,
                progress: this.progress,
                statusText: this.statusText,
                
                // LLM result
                llmResult: this.llmResult,
                llmLastPrompt: this.llmLastPrompt,
                
                // Dictation
                dictationText: this.dictationText,
                
                // Tab
                activeTab: this.activeTab,
                
                // Timestamp
                savedAt: Date.now()
            };
            
            try {
                sessionStorage.setItem('mywhisper_state', JSON.stringify(state));
            } catch (err) {
                console.warn('Failed to save state:', err);
            }
        },
        
        /**
         * Restore state from sessionStorage
         */
        restoreState() {
            try {
                const saved = sessionStorage.getItem('mywhisper_state');
                if (!saved) return false;
                
                const state = JSON.parse(saved);
                
                // Check if state is too old (more than 4 hours)
                const maxAge = 4 * 60 * 60 * 1000; // 4 hours in ms
                if (Date.now() - state.savedAt > maxAge) {
                    sessionStorage.removeItem('mywhisper_state');
                    return false;
                }
                
                // Restore results
                if (state.result) {
                    this.result = state.result;
                    this.resultMeta = state.resultMeta;
                    this.originalResult = state.originalResult;
                }
                
                // Restore speaker data
                if (state.detectedSpeakers?.length > 0) {
                    this.detectedSpeakers = state.detectedSpeakers;
                    this.speakerNames = state.speakerNames || {};
                    this.speakerSamples = state.speakerSamples || {};
                    this.sessionId = state.sessionId;
                    this.historyId = state.historyId;
                }
                
                // Restore client ID for result recovery
                if (state.clientId) {
                    this.clientId = state.clientId;
                }
                
                // Restore options
                if (state.options) {
                    this.options = { ...this.options, ...state.options };
                }
                
                // Restore file info (display only, not actual file)
                if (state.selectedFileName) {
                    // Create a fake file object for display purposes
                    this._restoredFileInfo = {
                        name: state.selectedFileName,
                        size: state.selectedFileSize
                    };
                }
                
                // Restore LLM result
                if (state.llmResult) {
                    this.llmResult = state.llmResult;
                    this.llmLastPrompt = state.llmLastPrompt || '';
                }
                
                // Restore dictation text
                if (state.dictationText) {
                    this.dictationText = state.dictationText;
                }
                
                // Restore tab
                if (state.activeTab) {
                    this.activeTab = state.activeTab;
                }
                
                // If was processing, restore the processing state immediately
                // This prevents the "server busy" message from appearing on refresh
                if (state.processing) {
                    this.processing = true;  // Set this immediately!
                    this.statusText = state.statusText || 'Reprise en cours...';
                    this.progress = state.progress || 0;
                    this.useServerProgress = true;
                }
                
                console.log('State restored from sessionStorage');
                return true;
                
            } catch (err) {
                console.warn('Failed to restore state:', err);
                sessionStorage.removeItem('mywhisper_state');
                return false;
            }
        },
        
        /**
         * Clear saved state
         */
        clearSavedState() {
            sessionStorage.removeItem('mywhisper_state');
        },
        
        /**
         * Get restored file info for display (when actual file is not available)
         */
        getFileDisplayInfo() {
            if (this.selectedFile) {
                return {
                    name: this.selectedFile.name,
                    size: this.selectedFile.size
                };
            }
            return this._restoredFileInfo || null;
        },
        
        /**
         * Restore speaker samples on server after page refresh.
         * If the server was restarted, it might have lost the speaker samples metadata
         * but the audio file might still exist.
         */
        async restoreSpeakerSamplesOnServer() {
            if (!this.sessionId || !this.speakerSamples || Object.keys(this.speakerSamples).length === 0) {
                return;
            }
            
            try {
                // First check if samples are available on server
                const checkResponse = await fetch(`/speaker-samples/${this.sessionId}`);
                
                if (checkResponse.ok) {
                    const data = await checkResponse.json();
                    // Check if server has the samples or just the file
                    if (!data.speaker_samples || Object.keys(data.speaker_samples).length === 0) {
                        // Server has file but no samples, restore them
                        await this.doRestoreSpeakerSamples();
                    } else {
                        console.log('Speaker samples already available on server');
                    }
                } else if (checkResponse.status === 404) {
                    // Session not found - try to restore
                    await this.doRestoreSpeakerSamples();
                }
            } catch (err) {
                console.warn('Failed to check/restore speaker samples on server:', err);
            }
        },
        
        /**
         * Actually send the restore request to server
         */
        async doRestoreSpeakerSamples() {
            try {
                const response = await fetch(`/speaker-samples/${this.sessionId}/restore`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ speaker_samples: this.speakerSamples })
                });
                
                if (response.ok) {
                    console.log('Speaker samples restored on server');
                } else {
                    // Audio file expired or not found
                    console.warn('Could not restore speaker samples - audio file may have expired');
                    // Clear the sample data since it's not usable
                    this.speakerSamples = {};
                    this.sessionId = null;
                    this.saveState();
                }
            } catch (err) {
                console.warn('Failed to restore speaker samples:', err);
            }
        },
        
        // ===== OLLAMA FUNCTIONS =====
        
        /**
         * Initialize - load saved settings
         */
        init() {
            this.loadSettings();
            // Load server settings (retention days)
            this.loadServerSettings();
            // Check Ollama status from backend (configured via .env)
            this.checkOllamaStatus();
            
            // Restore state from sessionStorage (before checking server status)
            const stateRestored = this.restoreState();
            
            // If restored to history tab, load history immediately
            if (this.activeTab === 'history') {
                this.loadHistory();
            }
            
            // If state was restored with speaker samples, ensure they're available on server
            if (stateRestored && this.sessionId && Object.keys(this.speakerSamples).length > 0) {
                this.restoreSpeakerSamplesOnServer();
            }
            
            // Check server status immediately
            this.checkServerStatus().then(() => {
                // If server is processing and we have restored state, resume tracking
                if (stateRestored && this.serverProcessing.is_processing && this.processing) {
                    // We were processing before refresh, resume progress tracking
                    console.log('Resuming progress tracking after refresh...');
                    this.resumeProgressTracking();
                } else if (stateRestored && !this.serverProcessing.is_processing && this.processing) {
                    // Server is not processing anymore but we thought we were processing
                    console.log('Server finished while we were away, recovering result...');
                    
                    if (this.result) {
                        // We have a result - show completed state
                        this.processing = false;
                        this.progress = 100;
                        this.statusText = '✅ Terminé !';
                    } else {
                        // No result - automatically fetch from history
                        this.fetchResultFromHistory();
                    }
                }
            });
            
            // Check server status periodically
            this.statusCheckInterval = setInterval(() => {
                this.checkServerStatus();
            }, 2000); // Check every 2 seconds
            
            // Save state periodically and on important changes
            setInterval(() => {
                if (this.result || this.processing || this.dictationText) {
                    this.saveState();
                }
            }, 5000); // Save every 5 seconds if there's something to save
            
            // Save state before page unload
            window.addEventListener('beforeunload', () => {
                this.saveState();
            });
        },
        
        /**
         * Try to recover result from history when connection was lost during processing
         */
        async tryRecoverFromHistory() {
            console.log('Trying to recover result from history...');
            
            try {
                // Get the most recent history item
                const response = await fetch('/history?limit=1&offset=0');
                if (!response.ok) {
                    throw new Error('Failed to fetch history');
                }
                
                const data = await response.json();
                
                if (data.transcriptions && data.transcriptions.length > 0) {
                    const lastItem = data.transcriptions[0];
                    
                    // Check if it was created recently (within last 30 minutes)
                    const createdAt = new Date(lastItem.created_at);
                    const now = new Date();
                    const ageMinutes = (now - createdAt) / (1000 * 60);
                    
                    if (ageMinutes < 30) {
                        // Recent transcription found - offer to view it
                        this.progress = 100;
                        this.statusText = '✅ Terminé ! Résultat disponible dans l\'historique';
                        this._recoveredHistoryId = lastItem.id;
                        
                        // Show notification and switch to history
                        setTimeout(() => {
                            if (confirm(`Le traitement a terminé pendant le rafraîchissement.\n\nFichier: ${lastItem.filename}\nLocuteurs: ${lastItem.speakers_count || 'N/A'}\n\nVoulez-vous voir le résultat dans l'historique ?`)) {
                                this.activeTab = 'history';
                                this.loadHistory();
                                this.viewHistoryItem(lastItem);
                            }
                            // Clear the stale state
                            this.clearSavedState();
                        }, 500);
                        
                        return;
                    }
                }
                
                // No recent result found - clear stale state
                this.progress = 0;
                this.statusText = '';
                this.clearSavedState();
                console.log('No recent history found - cleared stale state');
                
            } catch (err) {
                console.warn('Failed to recover from history:', err);
                this.progress = 0;
                this.statusText = '';
                this.clearSavedState();
            }
        },
        
        /**
         * Resume progress tracking after page refresh
         */
        resumeProgressTracking() {
            console.log('Resuming progress tracking...');
            this.useServerProgress = true;
            
            // Update from server status
            if (this.serverProcessing.progress_percent > 0) {
                this.progress = this.serverProcessing.progress_percent;
            }
            this.updateStatusFromServer(this.serverProcessing);
        },
        
        /**
         * Load server-side settings
         */
        async loadServerSettings() {
            try {
                const response = await fetch('/settings');
                if (response.ok) {
                    const data = await response.json();
                    this.retentionDays = data.retention_days || 90;
                }
            } catch (err) {
                console.error('Failed to load server settings:', err);
            }
        },
        
        /**
         * Save retention days setting
         */
        async saveRetentionDays() {
            this.retentionLoading = true;
            
            try {
                const response = await fetch('/settings/retention', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ days: parseInt(this.retentionDays) || 90 })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    this.retentionDays = data.retention_days;
                    alert('Durée de conservation mise à jour !');
                } else {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || 'Erreur lors de la sauvegarde');
                }
            } catch (err) {
                console.error('Failed to save retention days:', err);
                alert('Erreur: ' + err.message);
            } finally {
                this.retentionLoading = false;
            }
        },
        
        /**
         * Check if the server is currently processing something
         */
        async checkServerStatus() {
            try {
                const response = await fetch('/status');
                if (response.ok) {
                    const status = await response.json();
                    const wasProcessing = this.serverProcessing.is_processing;
                    this.serverProcessing = status;
                    
                    // Update progress from server if we're processing and server has progress info
                    if (this.processing && status.is_processing && status.progress_percent > 0) {
                        this.useServerProgress = true;
                        this.progress = status.progress_percent;
                        this.updateStatusFromServer(status);
                    }
                    
                    // Detect when server finishes processing (transition from true to false)
                    // and we're still showing processing state but don't have a result
                    if (wasProcessing && !status.is_processing && this.processing && !this.result) {
                        console.log('Server finished processing - fetching result from history');
                        this.fetchResultFromHistory();
                    }
                }
            } catch (err) {
                // Silent fail - server might be busy
                console.debug('Status check failed:', err);
            }
        },
        
        /**
         * Fetch the result using client ID cache or history as fallback
         */
        async fetchResultFromHistory() {
            this.statusText = '📥 Récupération du résultat...';
            console.log('Attempting to recover result...');
            
            try {
                // First, try to get result from server cache using client ID
                if (this.clientId) {
                    console.log('Trying to recover with client ID:', this.clientId);
                    const cacheResponse = await fetch(`/result/${this.clientId}`);
                    
                    if (cacheResponse.ok) {
                        const cached = await cacheResponse.json();
                        
                        if (cached.status === 'completed') {
                            console.log('Result recovered from server cache!');
                            
                            // Set the result
                            const result = cached.result;
                            this.result = cached.format === 'json' 
                                ? JSON.stringify(cached.content, null, 2)
                                : cached.content;
                            
                            this.resultMeta = {
                                language: result?.language,
                                duration: result?.duration
                            };
                            this.historyId = cached.history_id;
                            
                            // Extract speakers if diarization was enabled
                            if (result?.segments && result.segments.some(s => s.speaker)) {
                                this.extractSpeakers(result);
                                this.originalResult = this.result;
                            }
                            
                            this.processing = false;
                            this.progress = 100;
                            this.statusText = '✅ Terminé ! (récupéré)';
                            this.clientId = null; // Clear client ID after use
                            this.saveState();
                            
                            return;
                        } else if (cached.status === 'processing') {
                            console.log('Server still processing, resuming tracking...');
                            this.resumeProgressTracking();
                            return;
                        }
                    } else if (cacheResponse.status !== 404) {
                        console.warn('Unexpected response from cache:', cacheResponse.status);
                    }
                }
                
                // Fallback: Get result from history
                console.log('Falling back to history...');
                const response = await fetch('/history?limit=5&offset=0');
                if (!response.ok) throw new Error('Failed to fetch history');
                
                const data = await response.json();
                
                if (data.transcriptions && data.transcriptions.length > 0) {
                    // Get the file name we were processing
                    const expectedFileName = this.getFileDisplayInfo()?.name;
                    console.log('Looking for file:', expectedFileName);
                    
                    // Find a matching recent transcription
                    let bestMatch = null;
                    
                    for (const item of data.transcriptions) {
                        const createdAt = new Date(item.created_at);
                        const now = new Date();
                        const ageMinutes = (now - createdAt) / (1000 * 60);
                        
                        // Must be recent (within last 30 minutes)
                        if (ageMinutes > 30) continue;
                        
                        // If we know the filename, prefer exact match
                        if (expectedFileName && item.filename === expectedFileName) {
                            bestMatch = item;
                            console.log('Found exact filename match:', item.id, item.filename);
                            break;
                        }
                        
                        // Otherwise take the most recent one (first in list)
                        if (!bestMatch && ageMinutes < 5) {
                            bestMatch = item;
                            console.log('Found recent item:', item.id, item.filename);
                        }
                    }
                    
                    if (bestMatch) {
                        // Fetch full result
                        const fullResponse = await fetch(`/history/${bestMatch.id}`);
                        if (fullResponse.ok) {
                            const fullItem = await fullResponse.json();
                            
                            // Set the result
                            this.result = fullItem.result_text;
                            this.resultMeta = {
                                language: fullItem.language,
                                duration: fullItem.audio_duration
                            };
                            this.historyId = fullItem.id;
                            
                            // Extract speakers if diarization was enabled
                            if (fullItem.diarization && fullItem.result_json) {
                                this.extractSpeakers(fullItem.result_json);
                                this.originalResult = this.result;
                            }
                            
                            this.processing = false;
                            this.progress = 100;
                            this.statusText = '✅ Terminé ! (récupéré de l\'historique)';
                            this.clientId = null;
                            this.saveState();
                            
                            console.log('Result recovered from history:', bestMatch.id);
                            return;
                        }
                    }
                }
                
                // Fallback - no recent result found
                console.warn('No matching result found');
                this.processing = false;
                this.progress = 0;
                this.statusText = '⚠️ Résultat non trouvé - vérifiez l\'historique';
                this.clientId = null;
                this.clearSavedState();
                
            } catch (err) {
                console.error('Failed to recover result:', err);
                this.processing = false;
                this.progress = 0;
                this.statusText = '⚠️ Erreur de récupération';
                this.clientId = null;
                this.clearSavedState();
            }
        },
        
        /**
         * Update status text based on server progress info
         */
        updateStatusFromServer(status) {
            const step = status.current_step;
            const chunk = status.current_chunk;
            const total = status.total_chunks;
            const percent = status.progress_percent;
            
            switch (step) {
                case 'uploading':
                    this.statusText = '📤 Envoi du fichier...';
                    break;
                case 'transcribing':
                    if (status.audio_duration > 0) {
                        const mins = Math.floor(status.audio_duration / 60);
                        const secs = Math.round(status.audio_duration % 60);
                        if (mins > 0) {
                            this.statusText = `🎤 Transcription en cours... (${mins}m${secs}s d'audio)`;
                        } else {
                            this.statusText = `🎤 Transcription en cours... (${secs}s d'audio)`;
                        }
                    } else {
                        this.statusText = '🎤 Transcription en cours...';
                    }
                    break;
                case 'diarizing_loading':
                    this.statusText = '👥 Chargement du modèle de diarisation...';
                    break;
                case 'diarizing_processing':
                    // For short audio, show progress based on percent
                    if (percent >= 40 && percent < 85) {
                        const subProgress = Math.round(((percent - 40) / 45) * 100);
                        this.statusText = `👥 Analyse des voix... ${subProgress}%`;
                    } else {
                        this.statusText = '👥 Analyse des voix en cours...';
                    }
                    break;
                case 'diarizing_chunk':
                    if (total > 0) {
                        this.statusText = `👥 Diarisation: segment ${chunk}/${total}`;
                    } else {
                        this.statusText = '👥 Analyse des locuteurs...';
                    }
                    break;
                case 'diarizing_merging':
                    this.statusText = '👥 Harmonisation des locuteurs...';
                    break;
                case 'merging':
                    this.statusText = '🔗 Fusion transcription et locuteurs...';
                    break;
                case 'finalizing':
                    this.statusText = '📝 Finalisation...';
                    break;
                default:
                    // Keep current status text for unknown steps
                    break;
            }
        },
        
        /**
         * Check if we can start a new processing task
         * Returns true if available, false if blocked
         */
        canStartProcessing() {
            return !this.serverProcessing.is_processing && !this.processing && !this.dictationProcessing;
        },
        
        /**
         * Get blocking message if processing is blocked
         */
        getBlockingMessage() {
            if (this.serverProcessing.is_processing) {
                const type = this.serverProcessing.processing_type === 'dictation' ? 'une dictée' : 'un fichier';
                const file = this.serverProcessing.current_file || 'inconnu';
                return `Impossible de démarrer : ${type} est en cours de traitement (${file}). Veuillez attendre la fin de l'opération.`;
            }
            if (this.processing) {
                return 'Une transcription de fichier est en cours. Veuillez attendre.';
            }
            if (this.dictationProcessing) {
                return 'Une dictée est en cours de traitement. Veuillez attendre.';
            }
            return null;
        },
        
        /**
         * Request cancellation of current processing
         */
        async cancelProcessing() {
            if (!this.processing && !this.serverProcessing.is_processing) return;
            
            this.cancelRequested = true;
            this.statusText = '⏹️ Annulation en cours...';
            
            try {
                const response = await fetch('/cancel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    this.statusText = '⏹️ Annulation demandée, en attente...';
                } else {
                    this.cancelRequested = false;
                }
            } catch (err) {
                console.error('Cancel request failed:', err);
                this.cancelRequested = false;
            }
        },
        
        /**
         * Check Ollama status from backend
         */
        async checkOllamaStatus() {
            this.ollamaLoading = true;
            this.ollamaStatus = '';
            this.ollamaConnected = false;
            this.ollamaConfigured = false;
            
            try {
                const response = await fetch('/ollama/status');
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                
                this.ollamaConfigured = data.configured;
                this.ollamaConnected = data.connected;
                this.ollamaModels = data.models || [];
                this.ollamaStatus = data.message;
                
                // Use server-configured model or first available
                if (data.model) {
                    this.ollamaSettings.model = data.model;
                } else if (this.ollamaModels.length > 0 && !this.ollamaSettings.model) {
                    this.ollamaSettings.model = this.ollamaModels[0];
                }
                
            } catch (err) {
                console.error('Ollama status check error:', err);
                this.ollamaStatus = `Erreur: ${err.message}`;
                this.ollamaConnected = false;
                this.ollamaConfigured = false;
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
                this.error = 'Ollama non configuré ou non connecté. Vérifiez la configuration dans .env';
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
                
                // Call backend proxy endpoint
                const response = await fetch('/ollama/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.ollamaSettings.model,
                        prompt: fullPrompt,
                        stream: false
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP ${response.status}`);
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
         * Save settings to localStorage (only custom prompts, Ollama URL is in .env)
         */
        saveSettings() {
            const settings = {
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
                    if (settings.customPrompts && settings.customPrompts.length > 0) {
                        this.customPrompts = settings.customPrompts;
                    }
                }
            } catch (err) {
                console.error('Failed to load settings:', err);
            }
        },
        
        // ===== HISTORY FUNCTIONS =====
        
        /**
         * Load history from server
         */
        async loadHistory(offset = 0) {
            this.historyLoading = true;
            this.historyOffset = Math.max(0, offset);
            
            try {
                const response = await fetch(`/history?limit=${this.historyLimit}&offset=${this.historyOffset}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                
                // Add downloadFormat property to each item
                this.historyItems = data.transcriptions.map(item => ({
                    ...item,
                    downloadFormat: item.format || 'text'
                }));
                this.historyTotal = data.total;
                
            } catch (err) {
                console.error('Failed to load history:', err);
                this.historyItems = [];
                this.historyTotal = 0;
            } finally {
                this.historyLoading = false;
            }
        },
        
        /**
         * Format history date for display
         */
        formatHistoryDate(dateStr) {
            if (!dateStr) return '';
            
            const date = new Date(dateStr);
            const now = new Date();
            const diffMs = now - date;
            const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
            
            if (diffDays === 0) {
                return `Aujourd'hui à ${date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
            } else if (diffDays === 1) {
                return `Hier à ${date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
            } else if (diffDays < 7) {
                return `Il y a ${diffDays} jours`;
            } else {
                return date.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' });
            }
        },
        
        /**
         * Download a history item
         */
        async downloadHistoryItem(item) {
            try {
                const format = item.downloadFormat || 'text';
                const response = await fetch(`/history/${item.id}/download?format=${format}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const content = await response.text();
                
                // Determine file extension
                const extensions = { text: 'txt', json: 'json', srt: 'srt', vtt: 'vtt' };
                const ext = extensions[format] || 'txt';
                
                // Generate filename
                const baseName = item.filename.replace(/\.[^/.]+$/, '');
                const filename = `${baseName}.${ext}`;
                
                // Download
                const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
            } catch (err) {
                console.error('Download failed:', err);
                alert('Erreur lors du téléchargement');
            }
        },
        
        /**
         * View a history item in modal
         */
        async viewHistoryItem(item) {
            try {
                const response = await fetch(`/history/${item.id}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const fullItem = await response.json();
                this.historyViewItem = fullItem;
                
            } catch (err) {
                console.error('Failed to load item:', err);
                alert('Erreur lors du chargement');
            }
        },
        
        /**
         * Copy history item content to clipboard
         */
        async copyHistoryItem(item) {
            if (!item || !item.result_text) return;
            
            try {
                await navigator.clipboard.writeText(item.result_text);
                alert('Copié dans le presse-papier !');
            } catch (err) {
                console.error('Copy failed:', err);
            }
        },
        
        /**
         * Delete a history item
         */
        async deleteHistoryItem(item) {
            if (!confirm(`Supprimer la transcription "${item.filename}" ?`)) {
                return;
            }
            
            try {
                const response = await fetch(`/history/${item.id}`, {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                // Reload history
                await this.loadHistory(this.historyOffset);
                
            } catch (err) {
                console.error('Delete failed:', err);
                alert('Erreur lors de la suppression');
            }
        }
    };
}
