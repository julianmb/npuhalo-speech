#!/usr/bin/env python3
"""
server.py — OpenAI-Compatible Speech-to-Text & Speaker Diarization API Server + Web Studio
Optimized for AMD Strix Halo (XDNA 2 NPU + Zen 5 CPU / ROCm GPU).

Features:
- Live in-browser microphone recording with audio visualizer & timer.
- In-browser speaker renaming with instant subtitle / JSON export sync.
- Real-time transcript keyword search and speaker filtering pills.
- 150ms acoustic boundary padding and rolling prompt continuity.
"""

import os
import secrets
import tempfile
import argparse
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException, Depends, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import uvicorn

# Import pipeline logic
from transcribe_diarize import (
    process_pipeline,
    output_transcript,
    transcribe_audio_segment,
    ensure_npu_whisper,
    librosa_or_sf_load,
    format_timestamp,
    cyan, green, yellow, red, bold
)

app = FastAPI(
    title="Strix Halo Speech & Diarization Server",
    description="OpenAI-compatible Audio API accelerated on AMD XDNA 2 NPU and Zen 5 CPU/GPU",
    version="1.1.0"
)

# Enable CORS for external client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Configuration State
SERVER_CONFIG = {
    "api_key": None,
    "lemonade_url": "http://127.0.0.1:13305",
    "default_device": "cpu",
    "hf_token": None
}

# Serialize heavy pipeline work to protect the NPU/GPU from concurrent thrashing
PIPELINE_SEMAPHORE = threading.Semaphore(1)


def run_pipeline_blocking(fn, *args, **kwargs):
    """Run a pipeline function in the threadpool, gated by PIPELINE_SEMAPHORE."""
    def _worker():
        with PIPELINE_SEMAPHORE:
            return fn(*args, **kwargs)
    return run_in_threadpool(_worker)


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Enforce Bearer token authentication if API key is configured."""
    expected_key = SERVER_CONFIG["api_key"]
    if not expected_key:
        return True  # No auth required

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Missing Authorization header.", "type": "invalid_request_error", "code": "missing_api_key"}}
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}
        )
    return True


# ==============================================================================
# HTML Web UI Studio
# ==============================================================================
WEB_UI_HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strix Halo — Speech Diarization & ASR Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 50: '#f5f7ff', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' }
                    }
                }
            }
        }
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .speaker-0 { border-left-color: #3b82f6; background: rgba(59, 130, 246, 0.08); }
        .speaker-1 { border-left-color: #10b981; background: rgba(16, 185, 129, 0.08); }
        .speaker-2 { border-left-color: #8b5cf6; background: rgba(139, 92, 246, 0.08); }
        .speaker-3 { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.08); }
        .speaker-4 { border-left-color: #ec4899; background: rgba(236, 72, 153, 0.08); }
        @keyframes pulse-recording { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.05); } }
        .rec-active { animation: pulse-recording 1.5s infinite ease-in-out; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans antialiased">

    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                    <i class="fa-solid fa-microphone-lines text-white text-lg"></i>
                </div>
                <div>
                    <h1 class="font-bold text-lg leading-none tracking-tight flex items-center gap-2">
                        Strix Halo Speech Studio
                        <span class="text-xs font-mono font-normal bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded-full">XDNA 2 NPU</span>
                    </h1>
                    <p class="text-xs text-slate-400 mt-0.5">Whisper-v3-Turbo + Pyannote Diarization</p>
                </div>
            </div>

            <!-- API Key & Status -->
            <div class="flex items-center space-x-3">
                <div id="statusBadge" class="hidden sm:flex items-center gap-1.5 text-xs bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-3 py-1.5 rounded-lg">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    NPU Ready
                </div>
                <div class="relative flex items-center">
                    <i class="fa-solid fa-key absolute left-3 text-slate-500 text-xs"></i>
                    <input type="password" id="apiKeyInput" placeholder="Enter API Key..." 
                           class="bg-slate-800/80 border border-slate-700 text-xs rounded-lg pl-8 pr-8 py-1.5 w-48 sm:w-64 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-200 placeholder-slate-500 transition">
                    <button id="saveKeyBtn" title="Save API Key" class="absolute right-2 text-xs text-slate-400 hover:text-indigo-400">
                        <i class="fa-solid fa-check"></i>
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Left Column: Upload, Mic & Settings (5 cols) -->
        <div class="lg:col-span-5 space-y-6">
            
            <!-- Audio Input Box -->
            <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur space-y-4">
                <div class="flex items-center justify-between">
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                        <i class="fa-solid fa-cloud-arrow-up text-indigo-400"></i> Audio Input
                    </h2>
                    
                    <!-- Live Mic Record Button -->
                    <button id="micBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-2">
                        <i id="micIcon" class="fa-solid fa-microphone text-rose-400"></i>
                        <span id="micLabel">Record Mic</span>
                        <span id="recTimer" class="hidden font-mono text-[11px] text-rose-400 font-bold">00:00</span>
                    </button>
                </div>

                <!-- Dropzone -->
                <div id="dropZone" class="border-2 border-dashed border-slate-700 hover:border-indigo-500/50 bg-slate-800/30 hover:bg-slate-800/50 rounded-xl p-8 text-center cursor-pointer transition flex flex-col items-center justify-center">
                    <input type="file" id="audioFileInput" class="hidden" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac">
                    <div class="w-12 h-12 rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-3">
                        <i class="fa-solid fa-file-audio text-xl"></i>
                    </div>
                    <p class="text-sm font-medium text-slate-200">Click to upload or drag & drop</p>
                    <p class="text-xs text-slate-400 mt-1">WAV, MP3, M4A, FLAC, OGG, AAC (Up to 2GB)</p>
                </div>

                <!-- Selected File Display -->
                <div id="fileInfoCard" class="hidden p-3 bg-slate-800/60 border border-slate-700/60 rounded-xl flex items-center justify-between">
                    <div class="flex items-center space-x-3 overflow-hidden">
                        <i class="fa-solid fa-file-lines text-indigo-400 text-lg"></i>
                        <div class="truncate">
                            <p id="fileName" class="text-xs font-medium text-slate-200 truncate">audio.wav</p>
                            <p id="fileSize" class="text-[10px] text-slate-400">0 MB</p>
                        </div>
                    </div>
                    <button id="removeFileBtn" class="text-slate-400 hover:text-red-400 p-1">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>

                <!-- Audio Player Preview -->
                <div id="audioPlayerContainer" class="hidden">
                    <audio id="audioPreview" controls class="w-full h-10 rounded-lg outline-none"></audio>
                </div>
            </div>

            <!-- Pipeline Settings -->
            <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-indigo-400"></i> Pipeline Configuration
                </h2>

                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Diarization Device</label>
                        <select id="deviceSelect" class="w-full bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-indigo-500 outline-none">
                            <option value="cpu" selected>Zen 5 CPU (Default)</option>
                            <option value="rocm">Radeon GPU (ROCm)</option>
                            <option value="auto">Auto-Detect</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Num Speakers</label>
                        <input type="number" id="numSpeakersInput" placeholder="Auto" min="1" max="20"
                               class="w-full bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-indigo-500 outline-none">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 mb-1">Language</label>
                    <input type="text" id="langInput" placeholder="Auto-Detect (or 'en', 'es', 'zh', 'fr')"
                           class="w-full bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>

                <div class="pt-2">
                    <button id="processBtn" disabled class="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium text-sm rounded-xl shadow-lg shadow-indigo-600/20 disabled:shadow-none transition flex items-center justify-center gap-2">
                        <i class="fa-solid fa-bolt"></i>
                        <span>Start Diarization & Transcription</span>
                    </button>
                </div>
            </div>

            <!-- Hardware Topology Callout -->
            <div class="bg-gradient-to-br from-indigo-950/40 to-slate-900/60 border border-indigo-500/20 rounded-2xl p-5">
                <div class="flex items-start gap-3">
                    <i class="fa-solid fa-microchip text-indigo-400 text-lg mt-0.5"></i>
                    <div class="text-xs space-y-1">
                        <p class="font-semibold text-slate-200">Hardware Allocation</p>
                        <p class="text-slate-400">Whisper Turbo runs on the <span class="text-indigo-300">50 TOPS XDNA 2 NPU</span>. Diarization runs on <span class="text-indigo-300">Zen 5 CPU</span>. 150ms boundary padding active.</p>
                    </div>
                </div>
            </div>

        </div>

        <!-- Right Column: Results, Search, Speaker Renaming & Transcript (7 cols) -->
        <div class="lg:col-span-7 flex flex-col space-y-4">
            
            <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-xl flex-1 flex flex-col">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 mb-4 gap-3">
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                        <i class="fa-solid fa-align-left text-indigo-400"></i> Speaker-Attributed Transcript
                    </h2>

                    <!-- Export Actions -->
                    <div id="exportGroup" class="hidden flex items-center space-x-2">
                        <button id="copyBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-copy"></i> Copy
                        </button>
                        <button id="downloadSrtBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-download"></i> .SRT
                        </button>
                        <button id="downloadJsonBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-file-code"></i> JSON
                        </button>
                    </div>
                </div>

                <!-- Search and Filter Bar (Shown when results exist) -->
                <div id="filterBar" class="hidden flex flex-col sm:flex-row items-stretch sm:items-center gap-3 mb-4">
                    <div class="relative flex-1">
                        <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-slate-500 text-xs"></i>
                        <input type="text" id="searchInput" placeholder="Search transcript text..." 
                               class="w-full bg-slate-800/70 border border-slate-700 text-xs rounded-lg pl-8 pr-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    </div>
                    <div id="speakerFilterPills" class="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
                        <!-- Speaker filter pills rendered dynamically -->
                    </div>
                </div>

                <!-- Empty State -->
                <div id="emptyState" class="flex-1 flex flex-col items-center justify-center py-16 text-slate-500">
                    <i class="fa-solid fa-comments text-4xl mb-3 text-slate-700"></i>
                    <p class="text-sm">Upload or record audio to generate the speaker-attributed transcript.</p>
                </div>

                <!-- Loading State -->
                <div id="loadingState" class="hidden flex-1 flex flex-col items-center justify-center py-16">
                    <div class="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin mb-4"></div>
                    <p id="loadingMsg" class="text-sm font-medium text-slate-200">Processing Audio on AMD Strix Halo...</p>
                    <p class="text-xs text-slate-400 mt-1">Diarizing on CPU & transcribing turns on XDNA 2 NPU</p>
                </div>

                <!-- Transcript Output Container -->
                <div id="transcriptFeed" class="hidden space-y-3 max-h-[600px] overflow-y-auto pr-2">
                    <!-- Dynamic Turns Appended Here -->
                </div>

            </div>

        </div>

    </main>

    <script>
        // DOM Elements
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('audioFileInput');
        const fileInfoCard = document.getElementById('fileInfoCard');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const removeFileBtn = document.getElementById('removeFileBtn');
        const audioPlayerContainer = document.getElementById('audioPlayerContainer');
        const audioPreview = document.getElementById('audioPreview');
        const processBtn = document.getElementById('processBtn');
        const emptyState = document.getElementById('emptyState');
        const loadingState = document.getElementById('loadingState');
        const transcriptFeed = document.getElementById('transcriptFeed');
        const exportGroup = document.getElementById('exportGroup');
        const filterBar = document.getElementById('filterBar');
        const searchInput = document.getElementById('searchInput');
        const speakerFilterPills = document.getElementById('speakerFilterPills');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const saveKeyBtn = document.getElementById('saveKeyBtn');
        const copyBtn = document.getElementById('copyBtn');
        const downloadSrtBtn = document.getElementById('downloadSrtBtn');
        const downloadJsonBtn = document.getElementById('downloadJsonBtn');

        // Microphone Elements
        const micBtn = document.getElementById('micBtn');
        const micIcon = document.getElementById('micIcon');
        const micLabel = document.getElementById('micLabel');
        const recTimer = document.getElementById('recTimer');

        let selectedFile = null;
        let lastResponseData = null;
        let speakerNameMap = {};
        let activeSpeakerFilter = 'ALL';
        let mediaRecorder = null;
        let audioChunks = [];
        let timerInterval = null;
        let recordingSeconds = 0;

        // Init API Key from LocalStorage
        const savedKey = localStorage.getItem('strix_api_key');
        if (savedKey) { apiKeyInput.value = savedKey; }

        saveKeyBtn.addEventListener('click', () => {
            localStorage.setItem('strix_api_key', apiKeyInput.value.trim());
            saveKeyBtn.classList.add('text-emerald-400');
            setTimeout(() => saveKeyBtn.classList.remove('text-emerald-400'), 1500);
        });

        apiKeyInput.addEventListener('change', () => {
            localStorage.setItem('strix_api_key', apiKeyInput.value.trim());
        });

        // 1. Live Microphone Recording
        micBtn.addEventListener('click', async () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                // Stop recording
                mediaRecorder.stop();
                clearInterval(timerInterval);
                micBtn.classList.remove('bg-rose-600', 'text-white', 'rec-active');
                micBtn.classList.add('bg-slate-800', 'text-slate-200');
                micIcon.className = 'fa-solid fa-microphone text-rose-400';
                micLabel.textContent = 'Record Mic';
                recTimer.classList.add('hidden');
            } else {
                // Start recording
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioFile = new File([audioBlob], `mic_recording_${Date.now()}.wav`, { type: 'audio/wav' });
                        handleFile(audioFile);
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    recordingSeconds = 0;
                    recTimer.textContent = '00:00';
                    recTimer.classList.remove('hidden');
                    micBtn.classList.remove('bg-slate-800', 'text-slate-200');
                    micBtn.classList.add('bg-rose-600', 'text-white', 'rec-active');
                    micIcon.className = 'fa-solid fa-square';
                    micLabel.textContent = 'Stop';

                    timerInterval = setInterval(() => {
                        recordingSeconds++;
                        const m = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
                        const s = String(recordingSeconds % 60).padStart(2, '0');
                        recTimer.textContent = `${m}:${s}`;
                    }, 1000);

                } catch (err) {
                    alert('Microphone access denied: ' + err.message);
                }
            }
        });

        // File Selection Handlers
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-indigo-500'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-indigo-500'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-indigo-500');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });

        function handleFile(file) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
            fileInfoCard.classList.remove('hidden');
            processBtn.disabled = false;

            const objectUrl = URL.createObjectURL(file);
            audioPreview.src = objectUrl;
            audioPlayerContainer.classList.remove('hidden');
        }

        removeFileBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileInfoCard.classList.add('hidden');
            audioPlayerContainer.classList.add('hidden');
            audioPreview.src = '';
            processBtn.disabled = true;
        });

        // Process Request
        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            const apiKey = apiKeyInput.value.trim();
            const device = document.getElementById('deviceSelect').value;
            const numSpeakers = document.getElementById('numSpeakersInput').value;
            const language = document.getElementById('langInput').value;

            emptyState.classList.add('hidden');
            transcriptFeed.classList.add('hidden');
            exportGroup.classList.add('hidden');
            filterBar.classList.add('hidden');
            loadingState.classList.remove('hidden');
            processBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model', 'whisper-1');
            formData.append('response_format', 'verbose_json');
            formData.append('diarize', 'true');
            if (device) formData.append('device', device);
            if (numSpeakers) formData.append('num_speakers', numSpeakers);
            if (language) formData.append('language', language);

            try {
                const headers = {};
                if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;

                const res = await fetch('/v1/audio/transcriptions', {
                    method: 'POST',
                    headers: headers,
                    body: formData
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({ detail: res.statusText }));
                    throw new Error(err.detail?.error?.message || err.detail || 'HTTP Error ' + res.status);
                }

                const data = await res.json();
                lastResponseData = data;
                speakerNameMap = {};
                activeSpeakerFilter = 'ALL';
                renderTranscript();

            } catch (err) {
                alert('Transcription Failed: ' + err.message);
                emptyState.classList.remove('hidden');
            } finally {
                loadingState.classList.add('hidden');
                processBtn.disabled = false;
            }
        });

        function formatTime(sec) {
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            const ms = Math.floor((sec % 1) * 100);
            return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
        }

        const speakerColors = [
            'speaker-0 text-blue-400',
            'speaker-1 text-emerald-400',
            'speaker-2 text-violet-400',
            'speaker-3 text-amber-400',
            'speaker-4 text-pink-400'
        ];

        function getSpeakerStyle(speaker) {
            const num = parseInt((speaker || '').replace(/\D/g, '') || '0') % 5;
            return speakerColors[num];
        }

        function getDisplayName(rawSpeaker) {
            return speakerNameMap[rawSpeaker] || rawSpeaker;
        }

        window.renameSpeaker = function(rawSpeaker) {
            const current = getDisplayName(rawSpeaker);
            const newName = prompt(`Rename '${current}' to:`, current);
            if (newName && newName.trim()) {
                speakerNameMap[rawSpeaker] = newName.trim();
                renderTranscript();
            }
        };

        function renderFilterPills(speakers) {
            speakerFilterPills.innerHTML = '';
            
            // All pill
            const allBtn = document.createElement('button');
            allBtn.className = `text-[11px] px-2.5 py-1 rounded-full border transition whitespace-nowrap ${activeSpeakerFilter === 'ALL' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'}`;
            allBtn.textContent = 'All Speakers';
            allBtn.onclick = () => { activeSpeakerFilter = 'ALL'; renderTranscript(); };
            speakerFilterPills.appendChild(allBtn);

            speakers.forEach(spk => {
                const name = getDisplayName(spk);
                const btn = document.createElement('button');
                const isSelected = activeSpeakerFilter === spk;
                btn.className = `text-[11px] px-2.5 py-1 rounded-full border transition whitespace-nowrap ${isSelected ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'}`;
                btn.textContent = name;
                btn.onclick = () => { activeSpeakerFilter = spk; renderTranscript(); };
                speakerFilterPills.appendChild(btn);
            });
        }

        function renderTranscript() {
            if (!lastResponseData) return;
            const rawSegments = lastResponseData.segments || [];
            transcriptFeed.innerHTML = '';

            const allSpeakers = Array.from(new Set(rawSegments.map(s => s.speaker || 'SPEAKER_00')));
            renderFilterPills(allSpeakers);

            const searchQuery = searchInput.value.toLowerCase().trim();

            const filtered = rawSegments.filter(seg => {
                const matchesSpeaker = (activeSpeakerFilter === 'ALL') || (seg.speaker === activeSpeakerFilter);
                const matchesSearch = !searchQuery || seg.text.toLowerCase().includes(searchQuery) || getDisplayName(seg.speaker).toLowerCase().includes(searchQuery);
                return matchesSpeaker && matchesSearch;
            });

            if (filtered.length === 0) {
                transcriptFeed.innerHTML = '<div class="py-8 text-center text-xs text-slate-500">No matching speaker turns found.</div>';
            } else {
                filtered.forEach((seg) => {
                    const card = document.createElement('div');
                    const [styleClass, textColor] = getSpeakerStyle(seg.speaker).split(' ');
                    const displayName = getDisplayName(seg.speaker);
                    
                    let highlightedText = seg.text;
                    if (searchQuery) {
                        const regex = new RegExp(`(${searchQuery})`, 'gi');
                        highlightedText = seg.text.replace(regex, '<mark class="bg-indigo-500/30 text-indigo-200 px-1 rounded">$1</mark>');
                    }

                    card.className = `p-3.5 rounded-xl border-l-4 ${styleClass} border border-slate-800 transition`;
                    card.innerHTML = `
                        <div class="flex items-center justify-between mb-1">
                            <div class="flex items-center gap-2">
                                <button onclick="renameSpeaker('${seg.speaker}')" title="Click to Rename Speaker" class="text-xs font-semibold uppercase tracking-wider ${textColor} hover:underline flex items-center gap-1.5 cursor-pointer">
                                    <span>${displayName}</span>
                                    <i class="fa-solid fa-pen text-[9px] opacity-60"></i>
                                </button>
                                <button onclick="seekAudio(${seg.start})" class="text-[11px] font-mono text-slate-400 hover:text-indigo-400 flex items-center gap-1 cursor-pointer bg-slate-800/60 px-1.5 py-0.5 rounded">
                                    <i class="fa-solid fa-play text-[9px]"></i> ${formatTime(seg.start)} - ${formatTime(seg.end)}
                                </button>
                            </div>
                            ${seg.npu_latency_ms ? `<span class="text-[10px] text-slate-500">${seg.npu_latency_ms}ms NPU</span>` : ''}
                        </div>
                        <p class="text-xs text-slate-200 leading-relaxed">${highlightedText}</p>
                    `;
                    transcriptFeed.appendChild(card);
                });
            }

            transcriptFeed.classList.remove('hidden');
            exportGroup.classList.remove('hidden');
            filterBar.classList.remove('hidden');
        }

        searchInput.addEventListener('input', () => renderTranscript());

        window.seekAudio = function(seconds) {
            if (audioPreview && audioPreview.src) {
                audioPreview.currentTime = seconds;
                audioPreview.play();
            }
        };

        // Export Actions with Renamed Speakers
        copyBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            const text = (lastResponseData.segments || []).map(s => `[${formatTime(s.start)}] ${getDisplayName(s.speaker)}: ${s.text}`).join('\n');
            navigator.clipboard.writeText(text);
            copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            setTimeout(() => copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy', 1500);
        });

        downloadSrtBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            let srt = '';
            (lastResponseData.segments || []).forEach((s, idx) => {
                srt += `${idx + 1}\n${formatTime(s.start).replace('.', ',')}0 --> ${formatTime(s.end).replace('.', ',')}0\n${getDisplayName(s.speaker)}: ${s.text}\n\n`;
            });
            downloadFile(srt, 'transcript.srt', 'text/plain');
        });

        downloadJsonBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            const exportData = JSON.parse(JSON.stringify(lastResponseData));
            (exportData.segments || []).forEach(s => {
                s.speaker = getDisplayName(s.speaker);
            });
            downloadFile(JSON.stringify(exportData, null, 2), 'transcript.json', 'application/json');
        });

        function downloadFile(content, filename, type) {
            const blob = new Blob([content], { type: type });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def get_web_ui():
    """Serve the interactive Speech Studio Web UI."""
    return HTMLResponse(content=WEB_UI_HTML)


@app.get("/api")
def api_info():
    return {
        "name": "Strix Halo Heterogeneous Speech Server",
        "endpoints": {
            "web_ui": "/",
            "transcription": "/v1/audio/transcriptions",
            "models": "/v1/models",
            "health": "/health",
            "docs": "/docs"
        },
        "hardware": {
            "asr_engine": "OpenAI Whisper-v3-turbo on AMD XDNA 2 NPU (/dev/accel/accel0)",
            "diarization_engine": f"Pyannote Audio on {SERVER_CONFIG['default_device'].upper()}"
        }
    }


@app.get("/health")
def health_check():
    npu_ok = ensure_npu_whisper(SERVER_CONFIG["lemonade_url"])
    return {
        "status": "ok" if npu_ok else "warning",
        "npu_backend": "connected" if npu_ok else "unreachable",
        "lemonade_url": SERVER_CONFIG["lemonade_url"],
        "default_device": SERVER_CONFIG["default_device"]
    }


@app.get("/v1/models")
def list_models(_: bool = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "strix-halo-npu"
            },
            {
                "id": "whisper-v3-turbo",
                "object": "model",
                "created": 1700000000,
                "owned_by": "strix-halo-npu"
            },
            {
                "id": "whisper-v3-turbo-FLM",
                "object": "model",
                "created": 1700000000,
                "owned_by": "amd-xdna2"
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(0.0),
    diarize: bool = Form(True),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    device: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key)
):
    """
    OpenAI-compatible speech transcription & speaker diarization endpoint.
    """
    file_suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    active_device = device or SERVER_CONFIG["default_device"]

    try:
        if diarize:
            results = await run_pipeline_blocking(
                process_pipeline,
                audio_path=tmp_path,
                device=active_device,
                hf_token=SERVER_CONFIG["hf_token"],
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                lemonade_url=SERVER_CONFIG["lemonade_url"],
                language=language,
                padding_sec=0.15
            )

            full_text = " ".join(r["text"].strip() for r in results if r.get("text"))
            
            if response_format == "text":
                formatted_text = "\n".join(f"[{format_timestamp(r['start'])} -> {format_timestamp(r['end'])}] {r['speaker']}: {r['text']}" for r in results)
                return PlainTextResponse(formatted_text)

            elif response_format in ("srt", "vtt"):
                is_vtt = response_format == "vtt"
                subtitle_content = output_transcript(results, out_format=response_format)
                media_type = "text/vtt" if is_vtt else "application/x-subrip"
                return Response(content=subtitle_content, media_type=media_type)

            elif response_format == "verbose_json":
                y, sr = librosa_or_sf_load(tmp_path)
                duration = len(y) / sr
                
                segments = []
                for idx, r in enumerate(results):
                    segments.append({
                        "id": idx,
                        "seek": int(r["start"] * 100),
                        "start": r["start"],
                        "end": r["end"],
                        "text": r["text"],
                        "speaker": r["speaker"],
                        "tokens": [],
                        "temperature": temperature or 0.0,
                        "avg_logprob": -0.1,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.0,
                        "npu_latency_ms": r.get("latency_ms", 0.0)
                    })

                return JSONResponse({
                    "task": "transcribe",
                    "language": language or "en",
                    "duration": round(duration, 2),
                    "text": full_text,
                    "segments": segments
                })

            else:  # standard json
                return JSONResponse({
                    "text": full_text,
                    "speakers_detected": len(set(r["speaker"] for r in results)),
                    "turns": results
                })

        else:
            text, _ = await run_pipeline_blocking(
                transcribe_audio_segment,
                audio_path=tmp_path,
                lemonade_url=SERVER_CONFIG["lemonade_url"],
                language=language,
                prompt=prompt,
                fallback_device=active_device
            )

            if response_format == "text":
                return PlainTextResponse(text)
            elif response_format == "verbose_json":
                y, sr = librosa_or_sf_load(tmp_path)
                duration = len(y) / sr
                return JSONResponse({
                    "task": "transcribe",
                    "language": language or "en",
                    "duration": round(duration, 2),
                    "text": text,
                    "segments": [{"id": 0, "start": 0.0, "end": round(duration, 2), "text": text}]
                })
            else:
                return JSONResponse({"text": text})

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Strix Halo OpenAI-Compatible Speech & Diarization Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--api-key", "-k", type=str, default=None, help="API key for Bearer authentication (default: env API_KEY or auto-generated)")
    parser.add_argument("--no-auth", action="store_true", help="Disable API key authentication")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda", "rocm", "auto"], default="cpu", help="Default diarization device (default: cpu)")
    parser.add_argument("--lemonade-url", type=str, default=os.environ.get("LEMONADE_URL", "http://127.0.0.1:13305"), help="Lemonade NPU API URL")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for gated pyannote model")

    args = parser.parse_args()

    if args.no_auth:
        api_key = None
    else:
        api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = f"sk-strix-{secrets.token_hex(16)}"
            print(yellow(f"\n[!] No API key specified. Generated auto key:\n    {bold(api_key)}\n"))

    SERVER_CONFIG["api_key"] = api_key
    SERVER_CONFIG["lemonade_url"] = args.lemonade_url
    SERVER_CONFIG["default_device"] = args.device
    SERVER_CONFIG["hf_token"] = args.hf_token

    print("=" * 78)
    print(bold(" 🚀 STRIX HALO OPENAI-COMPATIBLE SPEECH SERVER + WEB STUDIO"))
    print("=" * 78)
    print(f"  • Web UI Studio:     {cyan(f'http://{args.host}:{args.port}/')}")
    print(f"  • API Key Required:  {green('Yes (Bearer Auth)') if api_key else yellow('Disabled (--no-auth)')}")
    if api_key:
        print(f"  • API Key:           {bold(api_key)}")
    print(f"  • ASR Model:         {green('Whisper-v3-turbo')} on AMD XDNA 2 NPU (/dev/accel/accel0)")
    print(f"  • Diarization:       {cyan('pyannote')} on {cyan(args.device.upper())}")
    print(f"  • Interactive Docs:  {cyan(f'http://{args.host}:{args.port}/docs')}")
    print("=" * 78 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
