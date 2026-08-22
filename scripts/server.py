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
import time
import secrets
import tempfile
import argparse
import threading
from pathlib import Path
from typing import Optional, Dict, Any

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

# Live job progress for client polling: {job_id: {"stage": ..., "detail": ..., "pct": ...}}
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


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
        @keyframes toast-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .seg-active { box-shadow: inset 0 0 0 1px rgb(99 102 241 / 0.6); }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen lg:h-screen lg:overflow-hidden flex flex-col font-sans antialiased">

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
                    <span id="statusDot" class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span id="statusText">NPU Ready</span>
                </div>
                <button id="lockBtn" title="Change API Key" aria-label="Change API Key"
                        class="text-xs bg-slate-800/80 border border-slate-700 text-slate-400 hover:text-indigo-400 px-3 py-1.5 rounded-lg transition flex items-center gap-1.5">
                    <i id="lockIcon" class="fa-solid fa-lock"></i>
                    <span class="hidden sm:inline">Locked</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 lg:min-h-0 lg:overflow-hidden">

        <!-- Left Column: Upload, Mic & Settings (5 cols) -->
        <div class="lg:col-span-5 space-y-6 lg:min-h-0 lg:overflow-y-auto lg:pr-1">
            
            <!-- Audio Input Box -->
            <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur space-y-4">
                <div class="flex items-center justify-between">
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                        <i class="fa-solid fa-cloud-arrow-up text-indigo-400"></i> Audio Input
                    </h2>
                    
                    <!-- Live Mic Record Button -->
                    <button id="micBtn" aria-label="Record from microphone" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-2">
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
                            <p class="text-[10px] text-slate-400"><span id="fileSize">0 MB</span><span id="fileDuration" class="hidden"> · <span class="font-mono"></span></span></p>
                        </div>
                    </div>
                    <button id="removeFileBtn" aria-label="Remove selected file" class="text-slate-400 hover:text-red-400 p-1">
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
                            <option value="cpu" selected>Zen 5 CPU</option>
                            <option value="rocm">Radeon GPU (ROCm)</option>
                            <option value="auto">Auto-Detect</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Transcription Engine</label>
                        <select id="asrSelect" class="w-full bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-indigo-500 outline-none">
                            <option value="auto" selected>Auto (NPU → Local)</option>
                            <option value="npu">NPU only (XDNA 2)</option>
                            <option value="cpu">Local CPU</option>
                            <option value="gpu">Local GPU (ROCm/CUDA)</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Num Speakers</label>
                        <input type="number" id="numSpeakersInput" placeholder="Auto" min="1" max="20"
                               class="w-full bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-indigo-500 outline-none">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Language</label>
                        <input type="text" id="langInput" placeholder="Auto-Detect"
                               class="w-full bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-indigo-500 outline-none">
                    </div>
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
        <div class="lg:col-span-7 flex flex-col space-y-4 lg:min-h-0">

            <div class="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 shadow-xl flex-1 flex flex-col lg:min-h-0">
                <div class="shrink-0 flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 mb-4 gap-3">
                    <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                        <i class="fa-solid fa-align-left text-indigo-400"></i> Speaker-Attributed Transcript
                    </h2>

                    <!-- Export Actions -->
                    <div id="exportGroup" class="hidden flex items-center space-x-2">
                        <button id="copyBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-copy"></i> Copy
                        </button>
                        <button id="downloadTxtBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-file-lines"></i> .TXT
                        </button>
                        <button id="downloadSrtBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-download"></i> .SRT
                        </button>
                        <button id="downloadVttBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-closed-captioning"></i> .VTT
                        </button>
                        <button id="downloadJsonBtn" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
                            <i class="fa-solid fa-file-code"></i> JSON
                        </button>
                    </div>
                </div>

                <!-- Search and Filter Bar (Shown when results exist) -->
                <div id="filterBar" class="hidden shrink-0 flex flex-col sm:flex-row items-stretch sm:items-center gap-3 mb-4">
                    <div class="relative flex-1">
                        <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-slate-500 text-xs"></i>
                        <input type="text" id="searchInput" placeholder="Search transcript text..."
                               class="w-full bg-slate-800/70 border border-slate-700 text-xs rounded-lg pl-8 pr-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                    </div>
                    <span id="matchCount" class="hidden text-[10px] font-mono text-slate-500 whitespace-nowrap"></span>
                    <div id="speakerFilterPills" class="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
                        <!-- Speaker filter pills rendered dynamically -->
                    </div>
                </div>

                <!-- Empty State -->
                <div id="emptyState" class="flex-1 flex flex-col items-center justify-center py-16 text-slate-500">
                    <i class="fa-solid fa-comments text-4xl mb-3 text-slate-700"></i>
                    <p class="text-sm">Upload or record audio to generate the speaker-attributed transcript.</p>
                    <div id="recentList" class="hidden w-full max-w-md mt-8 text-left">
                        <p class="text-[10px] uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-2">
                            <i class="fa-solid fa-clock-rotate-left"></i> Recent transcripts
                            <button id="clearHistoryBtn" class="ml-auto text-[10px] normal-case tracking-normal text-slate-500 hover:text-rose-400">Clear</button>
                        </p>
                        <div id="recentItems" class="space-y-1.5"></div>
                    </div>
                </div>

                <!-- Loading State: Progress Stepper -->
                <div id="loadingState" class="hidden flex-1 flex flex-col items-center justify-center py-12">
                    <div class="w-full max-w-md space-y-5">
                        <!-- Step 1: Upload -->
                        <div id="stepUpload" class="flex items-start gap-3">
                            <div class="stepDot w-7 h-7 shrink-0 rounded-full border-2 border-slate-700 bg-slate-800 flex items-center justify-center text-[10px] text-slate-500 transition-all duration-300">
                                <i class="fa-solid fa-arrow-up"></i>
                            </div>
                            <div class="flex-1 pt-0.5">
                                <div class="flex justify-between items-baseline">
                                    <span class="text-xs font-medium text-slate-400">Uploading audio</span>
                                    <span id="uploadPct" class="text-[10px] font-mono text-slate-500"></span>
                                </div>
                                <div class="h-1.5 bg-slate-800 rounded-full mt-1.5 overflow-hidden">
                                    <div id="uploadBar" class="h-full bg-indigo-500 w-0 transition-all duration-200"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Step 2: Diarization -->
                        <div id="stepDiarize" class="flex items-start gap-3">
                            <div class="stepDot w-7 h-7 shrink-0 rounded-full border-2 border-slate-700 bg-slate-800 flex items-center justify-center text-[10px] text-slate-500 transition-all duration-300">
                                <i class="fa-solid fa-users-viewfinder"></i>
                            </div>
                            <div class="flex-1 pt-0.5">
                                <div class="flex justify-between items-baseline">
                                    <span class="text-xs font-medium text-slate-400">Speaker diarization</span>
                                    <span id="diarizeDetail" class="text-[10px] font-mono text-slate-500"></span>
                                </div>
                                <p id="diarizeSub" class="text-[10px] text-slate-600 mt-1">Identifying who speaks when</p>
                            </div>
                        </div>

                        <!-- Step 3: Transcription -->
                        <div id="stepTranscribe" class="flex items-start gap-3">
                            <div class="stepDot w-7 h-7 shrink-0 rounded-full border-2 border-slate-700 bg-slate-800 flex items-center justify-center text-[10px] text-slate-500 transition-all duration-300">
                                <i class="fa-solid fa-feather"></i>
                            </div>
                            <div class="flex-1 pt-0.5">
                                <div class="flex justify-between items-baseline">
                                    <span class="text-xs font-medium text-slate-400">Transcribing turns</span>
                                    <span id="transcribeDetail" class="text-[10px] font-mono text-slate-500"></span>
                                </div>
                                <div class="h-1.5 bg-slate-800 rounded-full mt-1.5 overflow-hidden">
                                    <div id="transcribeBar" class="h-full bg-emerald-500 w-0 transition-all duration-300"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Elapsed -->
                        <div class="flex items-center justify-center gap-2 pt-2 text-[11px] text-slate-500">
                            <i class="fa-solid fa-stopwatch text-[9px]"></i>
                            <span id="elapsedTimer" class="font-mono">0s</span> elapsed
                            ·
                            <span id="engineLabel" class="text-slate-400">Auto engine</span>
                        </div>
                    </div>
                </div>

                <!-- Results Summary -->
                <div id="summaryStrip" class="hidden shrink-0 grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
                    <div class="bg-slate-800/40 border border-slate-800 rounded-xl px-3 py-2.5">
                        <p class="text-[9px] uppercase tracking-wider text-slate-500">Speakers</p>
                        <p id="statSpeakers" class="text-lg font-bold text-indigo-400 leading-tight">–</p>
                    </div>
                    <div class="bg-slate-800/40 border border-slate-800 rounded-xl px-3 py-2.5">
                        <p class="text-[9px] uppercase tracking-wider text-slate-500">Turns</p>
                        <p id="statTurns" class="text-lg font-bold text-slate-200 leading-tight">–</p>
                    </div>
                    <div class="bg-slate-800/40 border border-slate-800 rounded-xl px-3 py-2.5">
                        <p class="text-[9px] uppercase tracking-wider text-slate-500">Duration</p>
                        <p id="statDuration" class="text-lg font-bold text-slate-200 leading-tight">–</p>
                    </div>
                    <div class="bg-slate-800/40 border border-slate-800 rounded-xl px-3 py-2.5">
                        <p class="text-[9px] uppercase tracking-wider text-slate-500">Avg Latency</p>
                        <p id="statLatency" class="text-lg font-bold text-emerald-400 leading-tight">–</p>
                    </div>
                </div>

                <!-- Transcript Output Container -->
                <div id="transcriptFeed" class="hidden space-y-3 max-h-[600px] lg:max-h-none lg:flex-1 lg:min-h-0 overflow-y-auto pr-2">
                    <!-- Dynamic Turns Appended Here -->
                </div>

            </div>

        </div>

    </main>

    <!-- Auth Gate -->
    <div id="authGate" class="fixed inset-0 z-[200] bg-slate-950 flex items-center justify-center p-4">
        <div class="w-full max-w-sm bg-slate-900/60 border border-slate-800 rounded-2xl p-8 shadow-2xl text-center">
            <div class="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center mb-4 shadow-lg shadow-indigo-500/20">
                <i class="fa-solid fa-microphone-lines text-white text-xl"></i>
            </div>
            <h2 class="font-bold text-lg text-slate-100 tracking-tight">Strix Halo Speech Studio</h2>
            <p class="text-xs text-slate-400 mt-1 mb-6">Enter the server API key to unlock</p>
            <div class="relative">
                <i class="fa-solid fa-key absolute left-3 top-2.5 text-slate-500 text-xs"></i>
                <input type="password" id="gateKeyInput" placeholder="API Key" autocomplete="current-password"
                       class="w-full bg-slate-800/80 border border-slate-700 text-sm rounded-lg pl-8 pr-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            </div>
            <p id="gateError" class="hidden text-[11px] text-rose-400 mt-2"></p>
            <button id="gateUnlockBtn" class="w-full mt-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm rounded-xl shadow-lg shadow-indigo-600/20 transition flex items-center justify-center gap-2">
                <i class="fa-solid fa-unlock"></i> Unlock Studio
            </button>
        </div>
    </div>

    <!-- Rename Speaker Modal -->
    <div id="renameModal" class="hidden fixed inset-0 z-[90] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-sm shadow-2xl">
            <h3 class="text-sm font-semibold text-slate-200 mb-1">Rename Speaker</h3>
            <p id="renameRawLabel" class="text-[11px] font-mono text-slate-500 mb-4"></p>
            <input id="renameInput" type="text"
                   class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-4">
            <div class="flex justify-end gap-2">
                <button id="renameCancelBtn" class="text-xs px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 transition">Cancel</button>
                <button id="renameSaveBtn" class="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition">Save</button>
            </div>
        </div>
    </div>

    <!-- Toast Notifications -->
    <div id="toastContainer" class="fixed bottom-6 right-6 z-[100] space-y-2 max-w-sm"></div>

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
        const lockBtn = document.getElementById('lockBtn');
        const lockIcon = document.getElementById('lockIcon');
        const authGate = document.getElementById('authGate');
        const gateKeyInput = document.getElementById('gateKeyInput');
        const gateUnlockBtn = document.getElementById('gateUnlockBtn');
        const gateError = document.getElementById('gateError');
        const copyBtn = document.getElementById('copyBtn');
        const downloadSrtBtn = document.getElementById('downloadSrtBtn');
        const downloadJsonBtn = document.getElementById('downloadJsonBtn');
        const downloadTxtBtn = document.getElementById('downloadTxtBtn');
        const downloadVttBtn = document.getElementById('downloadVttBtn');
        const matchCount = document.getElementById('matchCount');
        const statusBadge = document.getElementById('statusBadge');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const elapsedTimer = document.getElementById('elapsedTimer');
        const engineLabel = document.getElementById('engineLabel');
        const stepUpload = document.getElementById('stepUpload');
        const stepDiarize = document.getElementById('stepDiarize');
        const stepTranscribe = document.getElementById('stepTranscribe');
        const uploadBar = document.getElementById('uploadBar');
        const uploadPct = document.getElementById('uploadPct');
        const diarizeDetail = document.getElementById('diarizeDetail');
        const diarizeSub = document.getElementById('diarizeSub');
        const transcribeBar = document.getElementById('transcribeBar');
        const transcribeDetail = document.getElementById('transcribeDetail');
        const summaryStrip = document.getElementById('summaryStrip');
        const statSpeakers = document.getElementById('statSpeakers');
        const statTurns = document.getElementById('statTurns');
        const statDuration = document.getElementById('statDuration');
        const statLatency = document.getElementById('statLatency');
        const recentList = document.getElementById('recentList');
        const recentItems = document.getElementById('recentItems');
        const clearHistoryBtn = document.getElementById('clearHistoryBtn');
        const renameModal = document.getElementById('renameModal');
        const renameRawLabel = document.getElementById('renameRawLabel');
        const renameInput = document.getElementById('renameInput');
        const renameSaveBtn = document.getElementById('renameSaveBtn');
        const renameCancelBtn = document.getElementById('renameCancelBtn');
        const toastContainer = document.getElementById('toastContainer');

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
        let currentObjectUrl = null;
        let renameTarget = null;
        let pollTimer = null;

        // ---------- Helpers ----------
        function escapeHtml(str) {
            return String(str ?? '')
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        }

        function escapeRegExp(str) {
            return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        }

        function showToast(message, type = 'error') {
            const toast = document.createElement('div');
            const styles = {
                error: 'bg-rose-950/90 border-rose-500/40 text-rose-200',
                success: 'bg-emerald-950/90 border-emerald-500/40 text-emerald-200',
                info: 'bg-slate-900/90 border-slate-600/50 text-slate-200'
            };
            const icons = { error: 'fa-circle-exclamation', success: 'fa-circle-check', info: 'fa-circle-info' };
            toast.className = `flex items-start gap-2 text-xs px-4 py-3 rounded-xl border shadow-xl backdrop-blur max-w-full ${styles[type] || styles.info} animate-[toast-in_.2s_ease-out]`;
            toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info} mt-0.5"></i><span class="leading-relaxed">${escapeHtml(message)}</span>`;
            toastContainer.appendChild(toast);
            setTimeout(() => {
                toast.style.transition = 'opacity .3s';
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, 4500);
        }

        function setAudioSource(file) {
            if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
            currentObjectUrl = URL.createObjectURL(file);
            audioPreview.src = currentObjectUrl;
        }

        // Live NPU status badge
        async function refreshStatus() {
            try {
                const res = await fetch('/health');
                const h = await res.json();
                if (h.npu_backend === 'connected') {
                    statusDot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
                    statusText.textContent = 'NPU Ready';
                    statusBadge.className = statusBadge.className.replace(/bg-\S+\/10 border-\S+\/20 text-\S+/, 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400');
                } else {
                    statusDot.className = 'w-2 h-2 rounded-full bg-amber-400';
                    statusText.textContent = 'NPU Offline · Local Fallback';
                    statusBadge.className = statusBadge.className.replace(/bg-\S+\/10 border-\S+\/20 text-\S+/, 'bg-amber-500/10 border-amber-500/20 text-amber-400');
                }
            } catch {
                statusDot.className = 'w-2 h-2 rounded-full bg-rose-500';
                statusText.textContent = 'Server Unreachable';
            }
        }
        refreshStatus();
        setInterval(refreshStatus, 15000);

        // Convert recorded audio (webm/ogg) into a real 16-bit PCM WAV file
        async function convertToWavFile(blob, filename) {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            try {
                const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
                const numCh = Math.min(buf.numberOfChannels, 2);
                const len = buf.length;
                const wav = new ArrayBuffer(44 + len * numCh * 2);
                const view = new DataView(wav);
                const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
                writeStr(0, 'RIFF'); view.setUint32(4, 36 + len * numCh * 2, true); writeStr(8, 'WAVE');
                writeStr(12, 'fmt '); view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); view.setUint16(22, numCh, true);
                view.setUint32(24, buf.sampleRate, true); view.setUint32(28, buf.sampleRate * numCh * 2, true);
                view.setUint16(32, numCh * 2, true); view.setUint16(34, 16, true);
                writeStr(36, 'data'); view.setUint32(40, len * numCh * 2, true);
                const chans = [];
                for (let c = 0; c < numCh; c++) chans.push(buf.getChannelData(c));
                let off = 44;
                for (let i = 0; i < len; i++) {
                    for (let c = 0; c < numCh; c++) {
                        const v = Math.max(-1, Math.min(1, chans[c][i]));
                        view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7FFF, true);
                        off += 2;
                    }
                }
                return new File([wav], filename, { type: 'audio/wav' });
            } finally {
                ctx.close();
            }
        }

        // ---------- Auth Gate ----------
        const getApiKey = () => localStorage.getItem('strix_api_key') || '';

        function showGate(message) {
            gateError.textContent = message || '';
            gateError.classList.toggle('hidden', !message);
            authGate.classList.remove('hidden');
            gateKeyInput.value = '';
            setTimeout(() => gateKeyInput.focus(), 50);
        }

        function hideGate() {
            authGate.classList.add('hidden');
            lockIcon.className = 'fa-solid fa-lock-open';
            lockBtn.querySelector('span').textContent = 'Unlocked';
        }

        async function validateKey(key) {
            try {
                const res = await fetch('/v1/models', { headers: { 'Authorization': 'Bearer ' + key } });
                return res.ok;
            } catch {
                return false;
            }
        }

        async function initAuth() {
            const key = getApiKey();
            if (key && await validateKey(key)) {
                hideGate();
            } else {
                if (key) localStorage.removeItem('strix_api_key');
                showGate();
            }
        }
        initAuth();

        async function attemptUnlock() {
            const key = gateKeyInput.value.trim();
            if (!key) {
                gateError.textContent = 'Please enter the API key.';
                gateError.classList.remove('hidden');
                return;
            }
            gateUnlockBtn.disabled = true;
            const ok = await validateKey(key);
            gateUnlockBtn.disabled = false;
            if (ok) {
                localStorage.setItem('strix_api_key', key);
                hideGate();
                showToast('Studio unlocked.', 'success');
                refreshStatus();
            } else {
                showGate('Invalid API key. Check the value printed when the server started.');
            }
        }

        gateUnlockBtn.addEventListener('click', attemptUnlock);
        gateKeyInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') attemptUnlock(); });
        lockBtn.addEventListener('click', () => showGate());

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

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks);
                        try {
                            const wavFile = await convertToWavFile(audioBlob, `mic_recording_${Date.now()}.wav`);
                            handleFile(wavFile);
                        } catch (err) {
                            showToast('Could not process recording: ' + err.message);
                        }
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
                    showToast('Microphone access denied: ' + err.message);
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

        const MAX_FILE_BYTES = 500 * 1024 * 1024; // 500 MB soft limit (server also enforces)
        const MAX_DURATION_SEC = 3 * 3600; // 3 h

        function formatDuration(sec) {
            if (!isFinite(sec) || sec <= 0) return '';
            const h = Math.floor(sec / 3600);
            const m = Math.floor((sec % 3600) / 60);
            const s = Math.floor(sec % 60);
            return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
        }

        function handleFile(file) {
            if (file.size > MAX_FILE_BYTES) {
                showToast(`File too large (${(file.size / (1024*1024)).toFixed(0)} MB) — limit is ${MAX_FILE_BYTES / (1024*1024)} MB.`, 'error');
                return;
            }
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
            const durWrap = document.getElementById('fileDuration');
            durWrap.classList.add('hidden');
            fileInfoCard.classList.remove('hidden');
            processBtn.disabled = false;

            setAudioSource(file);
            audioPlayerContainer.classList.remove('hidden');

            // Duration preview once metadata is available
            const onMeta = () => {
                const d = audioPreview.duration;
                if (isFinite(d) && d > 0) {
                    durWrap.querySelector('span').textContent = formatDuration(d);
                    durWrap.classList.remove('hidden');
                    if (d > MAX_DURATION_SEC) {
                        showToast(`Long audio (${formatDuration(d)}) — transcription may take several minutes.`, 'info');
                    }
                }
                audioPreview.removeEventListener('loadedmetadata', onMeta);
            };
            audioPreview.addEventListener('loadedmetadata', onMeta);
        }

        removeFileBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileInfoCard.classList.add('hidden');
            audioPlayerContainer.classList.add('hidden');
            if (currentObjectUrl) { URL.revokeObjectURL(currentObjectUrl); currentObjectUrl = null; }
            audioPreview.src = '';
            processBtn.disabled = true;
        });

        // ---------- Recent Transcript History ----------
        const HISTORY_KEY = 'strix_history';
        const getHistory = () => { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; } };
        const saveHistory = (h) => localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, 5)));

        function renderHistory() {
            const hist = getHistory();
            recentItems.innerHTML = '';
            if (!hist.length) {
                recentList.classList.add('hidden');
                return;
            }
            hist.forEach((item, i) => {
                const row = document.createElement('button');
                row.className = 'w-full flex items-center gap-2 px-3 py-2 bg-slate-800/40 border border-slate-800 rounded-lg text-left hover:border-indigo-500/50 transition group';
                row.innerHTML = `
                    <i class="fa-solid fa-file-audio text-[10px] text-slate-500 group-hover:text-indigo-400"></i>
                    <span class="text-[11px] text-slate-300 truncate flex-1">${escapeHtml(item.name)}</span>
                    <span class="text-[9px] font-mono text-slate-500 whitespace-nowrap">${escapeHtml(item.when)} · ${item.turns} turns</span>
                    <span class="text-[9px] text-indigo-400 opacity-0 group-hover:opacity-100 transition">load</span>`;
                row.onclick = () => restoreHistory(i);
                recentItems.appendChild(row);
            });
            recentList.classList.remove('hidden');
        }

        function rememberTranscript(name, data) {
            const hist = getHistory();
            hist.unshift({
                name,
                when: new Date().toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
                turns: (data.segments || []).length,
                data
            });
            saveHistory(hist);
            renderHistory();
        }

        function restoreHistory(idx) {
            const item = getHistory()[idx];
            if (!item) return;
            lastResponseData = item.data;
            speakerNameMap = {};
            activeSpeakerFilter = 'ALL';
            searchInput.value = '';
            emptyState.classList.add('hidden');
            transcriptFeed.classList.remove('hidden');
            exportGroup.classList.remove('hidden');
            filterBar.classList.remove('hidden');
            renderTranscript();
            renderSummary(lastResponseData);
            showToast(`Restored "${item.name}" (${item.turns} turns).`, 'info');
        }

        clearHistoryBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            localStorage.removeItem(HISTORY_KEY);
            renderHistory();
            showToast('History cleared.', 'info');
        });

        renderHistory();

        // ---------- Progress Stepper ----------
        function setStep(stepEl, state) {
            const dot = stepEl.querySelector('.stepDot');
            const icon = dot.querySelector('i');
            const label = stepEl.querySelector('span');
            dot.classList.remove('border-slate-700', 'bg-slate-800', 'text-slate-500',
                                 'border-indigo-500', 'text-indigo-400', 'animate-pulse',
                                 '!bg-emerald-500/10', '!border-emerald-500', '!text-emerald-400');
            label.classList.remove('text-slate-400', 'text-slate-200', 'text-slate-500');
            if (state === 'active') {
                dot.classList.add('border-indigo-500', 'text-indigo-400', 'animate-pulse');
                label.classList.add('text-slate-200');
            } else if (state === 'done') {
                dot.classList.add('!bg-emerald-500/10', '!border-emerald-500', '!text-emerald-400');
                label.classList.add('text-slate-400');
                icon.className = 'fa-solid fa-check';
            } else {
                dot.classList.add('border-slate-700', 'bg-slate-800', 'text-slate-500');
                label.classList.add('text-slate-500');
            }
        }

        function resetSteps() {
            setStep(stepUpload, 'pending');
            setStep(stepDiarize, 'pending');
            setStep(stepTranscribe, 'pending');
            uploadBar.style.width = '0%';
            uploadPct.textContent = '';
            diarizeDetail.textContent = '';
            transcribeDetail.textContent = '';
            transcribeBar.style.width = '0%';
            // Restore original icons
            stepUpload.querySelector('i').className = 'fa-solid fa-arrow-up';
            stepDiarize.querySelector('i').className = 'fa-solid fa-users-viewfinder';
            stepTranscribe.querySelector('i').className = 'fa-solid fa-feather';
        }

        function applyProgress(p) {
            if (!p || !p.stage) return;
            if (p.stage === 'load' || p.stage === 'diarize') {
                setStep(stepUpload, 'done');
                setStep(stepDiarize, 'active');
                diarizeDetail.textContent = p.detail || '';
                if (p.stage === 'load') diarizeSub.textContent = p.detail || '';
                else diarizeSub.textContent = 'Identifying who speaks when';
            } else if (p.stage === 'transcribe') {
                setStep(stepUpload, 'done');
                setStep(stepDiarize, 'done');
                setStep(stepTranscribe, 'active');
                transcribeDetail.textContent = p.detail || '';
                if (typeof p.pct === 'number') transcribeBar.style.width = p.pct + '%';
            } else if (p.stage === 'done') {
                setStep(stepUpload, 'done');
                setStep(stepDiarize, 'done');
                setStep(stepTranscribe, 'done');
                transcribeBar.style.width = '100%';
            }
        }

        function startProgressPolling(jobId) {
            const poll = async () => {
                try {
                    const res = await fetch('/v1/progress/' + jobId, { headers: { 'Authorization': 'Bearer ' + getApiKey() } });
                    if (res.ok) applyProgress(await res.json());
                } catch { /* transient */ }
            };
            pollTimer = setInterval(poll, 1000);
            poll();
        }

        function stopProgressPolling() {
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }

        // Process Request
        function transcribeWithProgress(formData, headers) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/v1/audio/transcriptions');
                if (headers.Authorization) xhr.setRequestHeader('Authorization', headers.Authorization);
                xhr.upload.onprogress = (e) => {
                    if (!e.lengthComputable) return;
                    const pct = Math.round((e.loaded / e.total) * 100);
                    uploadBar.style.width = pct + '%';
                    uploadPct.textContent = pct >= 100 ? 'done' : pct + '%';
                    if (pct >= 100) setStep(stepUpload, 'done');
                };
                xhr.onload = () => {
                    let data = null;
                    try { data = JSON.parse(xhr.responseText); } catch (e) { /* non-JSON */ }
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(data);
                    } else {
                        let msg = data?.detail?.error?.message || data?.detail || ('HTTP Error ' + xhr.status);
                        if (xhr.status === 401) {
                            msg = 'Unauthorized — enter your API key in the field at the top right, then try again.';
                        }
                        reject(new Error(msg));
                    }
                };
                xhr.onerror = () => reject(new Error('Network error while contacting the server.'));
                xhr.send(formData);
            });
        }

        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            const apiKey = getApiKey();
            if (!apiKey) {
                showGate('Enter your API key to start.');
                return;
            }
            const device = document.getElementById('deviceSelect').value;
            const asrBackend = document.getElementById('asrSelect').value;
            const numSpeakers = document.getElementById('numSpeakersInput').value;
            const language = document.getElementById('langInput').value;
            let jobId;
            try { jobId = crypto.randomUUID(); } catch { jobId = 'job-' + Date.now() + '-' + Math.random().toString(36).slice(2); }

            emptyState.classList.add('hidden');
            transcriptFeed.classList.add('hidden');
            exportGroup.classList.add('hidden');
            filterBar.classList.add('hidden');
            summaryStrip.classList.add('hidden');
            loadingState.classList.remove('hidden');
            resetSteps();
            processBtn.disabled = true;

            const engineNames = { auto: 'Auto engine', npu: 'NPU (XDNA 2)', cpu: 'Local CPU', gpu: 'Local GPU' };
            engineLabel.textContent = engineNames[asrBackend] || 'Auto engine';

            const procStart = Date.now();
            elapsedTimer.textContent = '0s';
            const elapsedInterval = setInterval(() => {
                elapsedTimer.textContent = Math.floor((Date.now() - procStart) / 1000) + 's';
            }, 1000);

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model', 'whisper-1');
            formData.append('response_format', 'verbose_json');
            formData.append('diarize', 'true');
            formData.append('asr_backend', asrBackend);
            formData.append('job_id', jobId);
            if (device) formData.append('device', device);
            if (numSpeakers) formData.append('num_speakers', numSpeakers);
            if (language) formData.append('language', language);

            const headers = {};
            headers['Authorization'] = 'Bearer ' + apiKey;
            let recovering = false;

            startProgressPolling(jobId);

            try {
                const data = await transcribeWithProgress(formData, headers);
                lastResponseData = data;
                speakerNameMap = {};
                activeSpeakerFilter = 'ALL';
                renderTranscript();
                renderSummary(data);
                rememberTranscript(selectedFile.name, data);
                showToast('Transcription complete: ' + (data.segments || []).length + ' speaker turns.', 'success');

            } catch (err) {
                if (String(err.message).includes('Unauthorized')) {
                    stopProgressPolling();
                    localStorage.removeItem('strix_api_key');
                    showGate('Your session was rejected — the API key may have changed. Enter it again to unlock.');
                } else if (String(err.message).includes('Network error') && selectedFile) {
                    // POST connection dropped, but the server may still be processing
                    // (or already finished). Try to recover the result via polling.
                    stopProgressPolling();
                    recovering = true;
                    recoverJob(jobId);
                } else {
                    stopProgressPolling();
                    showToast('Transcription failed: ' + err.message);
                    emptyState.classList.remove('hidden');
                }
            } finally {
                clearInterval(elapsedInterval);
                if (!recovering) loadingState.classList.add('hidden');
                processBtn.disabled = false;
            }
        });

        // Recover a job whose POST connection was lost mid-flight.
        function recoverJob(jobId) {
            loadingState.classList.remove('hidden');
            showToast('Connection lost — the server keeps working. Waiting for the result…', 'info');
            let misses = 0;
            const startedAt = Date.now();
            const recoverTimer = setInterval(async () => {
                try {
                    const res = await fetch('/v1/progress/' + jobId, { headers: { 'Authorization': 'Bearer ' + getApiKey() } });
                    if (!res.ok) throw new Error('poll failed');
                    misses = 0;
                    const p = await res.json();
                    applyProgress(p);
                    if (p.stage === 'done' && p.result) {
                        clearInterval(recoverTimer);
                        lastResponseData = p.result;
                        speakerNameMap = {};
                        activeSpeakerFilter = 'ALL';
                        renderTranscript();
                        renderSummary(p.result);
                        rememberTranscript(selectedFile ? selectedFile.name : 'recording', p.result);
                        loadingState.classList.add('hidden');
                        showToast('Recovered transcript after connection drop.', 'success');
                    } else if (Date.now() - startedAt > 20 * 60 * 1000) {
                        clearInterval(recoverTimer);
                        loadingState.classList.add('hidden');
                        emptyState.classList.remove('hidden');
                        showToast('Gave up waiting for the result after 20 minutes.', 'error');
                    }
                } catch {
                    misses++;
                    if (misses > 8) {
                        clearInterval(recoverTimer);
                        loadingState.classList.add('hidden');
                        emptyState.classList.remove('hidden');
                        showToast('Lost contact with the server. If it finishes the job, check Recent transcripts later.', 'error');
                    }
                }
            }, 3000);
        }

        function renderSummary(data) {
            const segs = data.segments || [];
            if (!segs.length) return;
            const speakers = new Set(segs.map(s => s.speaker || 'SPEAKER_00')).size;
            const duration = Math.max(...segs.map(s => s.end || 0));
            const latencies = segs.map(s => s.npu_latency_ms).filter(v => typeof v === 'number' && v > 0);
            const avg = latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : null;

            statSpeakers.textContent = speakers;
            statTurns.textContent = segs.length;
            statDuration.textContent = duration >= 60
                ? Math.floor(duration / 60) + 'm ' + Math.round(duration % 60) + 's'
                : Math.round(duration) + 's';
            statLatency.textContent = avg !== null ? avg + 'ms' : '–';

            summaryStrip.classList.remove('hidden');
        }

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
            renameTarget = rawSpeaker;
            renameRawLabel.textContent = rawSpeaker;
            renameInput.value = getDisplayName(rawSpeaker);
            renameModal.classList.remove('hidden');
            renameInput.focus();
            renameInput.select();
        };

        function closeRenameModal() {
            renameModal.classList.add('hidden');
            renameTarget = null;
        }

        renameCancelBtn.addEventListener('click', closeRenameModal);
        renameModal.addEventListener('click', (e) => { if (e.target === renameModal) closeRenameModal(); });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !renameModal.classList.contains('hidden')) closeRenameModal();
        });
        renameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') renameSaveBtn.click();
        });

        renameSaveBtn.addEventListener('click', () => {
            const newName = renameInput.value.trim();
            if (newName && renameTarget) {
                speakerNameMap[renameTarget] = newName;
                renderTranscript();
                showToast('Speaker renamed to "' + newName + '".', 'success');
            }
            closeRenameModal();
        });

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

            if (searchQuery || activeSpeakerFilter !== 'ALL') {
                matchCount.textContent = filtered.length + ' / ' + rawSegments.length + ' turns';
                matchCount.classList.remove('hidden');
            } else {
                matchCount.classList.add('hidden');
            }

            if (filtered.length === 0) {
                transcriptFeed.innerHTML = '<div class="py-8 text-center text-xs text-slate-500">No matching speaker turns found.</div>';
            } else {
                filtered.forEach((seg) => {
                    const card = document.createElement('div');
                    const [styleClass, textColor] = getSpeakerStyle(seg.speaker).split(' ');
                    const displayName = getDisplayName(seg.speaker);

                    const safeText = escapeHtml(seg.text);
                    let highlightedText = safeText;
                    if (searchQuery) {
                        const idx = seg.text.toLowerCase().indexOf(searchQuery);
                        if (idx !== -1) {
                            highlightedText = escapeHtml(seg.text.slice(0, idx)) +
                                '<mark class="bg-indigo-500/30 text-indigo-200 px-1 rounded">' +
                                escapeHtml(seg.text.slice(idx, idx + searchQuery.length)) + '</mark>' +
                                escapeHtml(seg.text.slice(idx + searchQuery.length));
                        }
                    }

                    card.className = `p-3.5 rounded-xl border-l-4 ${styleClass} border border-slate-800 transition`;
                    card.dataset.start = seg.start;
                    card.dataset.end = seg.end;
                    card.innerHTML = `
                        <div class="flex items-center justify-between mb-1 flex-wrap gap-y-1 gap-x-2">
                            <div class="flex items-center gap-2 flex-wrap min-w-0">
                                <button onclick="renameSpeaker('${escapeHtml(seg.speaker)}')" title="Click to Rename Speaker" class="text-xs font-semibold uppercase tracking-wider ${textColor} hover:underline flex items-center gap-1.5 cursor-pointer">
                                    <span>${escapeHtml(displayName)}</span>
                                    <i class="fa-solid fa-pen text-[9px] opacity-60"></i>
                                </button>
                                <button onclick="seekAudio(${Number(seg.start) || 0})" class="text-[11px] font-mono text-slate-400 hover:text-indigo-400 flex items-center gap-1 cursor-pointer bg-slate-800/60 px-1.5 py-0.5 rounded whitespace-nowrap">
                                    <i class="fa-solid fa-play text-[9px]"></i> ${formatTime(seg.start)} - ${formatTime(seg.end)}
                                </button>
                                ${seg.backend ? `<span class="text-[9px] font-mono px-1.5 py-0.5 rounded border whitespace-nowrap ${String(seg.backend).includes('NPU') ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-slate-800 text-slate-400 border-slate-700'}">${escapeHtml(seg.backend)}</span>` : ''}
                            </div>
                            ${seg.npu_latency_ms ? `<span class="text-[10px] text-slate-500 whitespace-nowrap">${seg.npu_latency_ms}ms</span>` : ''}
                        </div>
                        <p class="text-xs text-slate-200 leading-relaxed break-words">${highlightedText}</p>
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

        // Highlight + scroll to the segment currently playing
        let activeSegCard = null;
        audioPreview.addEventListener('timeupdate', () => {
            if (!lastResponseData) return;
            const t = audioPreview.currentTime;
            const segs = lastResponseData.segments || [];
            const active = segs.find(s => t >= s.start && t < s.end);
            if (!active) return;
            const card = Array.from(transcriptFeed.children).find(el => {
                const s = parseFloat(el.dataset.start), e2 = parseFloat(el.dataset.end);
                return !isNaN(s) && !isNaN(e2) && t >= s && t < e2;
            });
            if (card && card !== activeSegCard) {
                if (activeSegCard) activeSegCard.classList.remove('seg-active');
                activeSegCard = card;
                card.classList.add('seg-active');
                card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
        });

        function fmtStamp(sec, comma) {
            const h = Math.floor(sec / 3600);
            const m = Math.floor((sec % 3600) / 60);
            const s = Math.floor(sec % 60);
            const ms = Math.round((sec - Math.floor(sec)) * 1000);
            const base = `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}${comma ? ',' : '.'}${String(ms).padStart(3,'0')}`;
            return base;
        }

        // Export Actions with Renamed Speakers
        copyBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            const text = (lastResponseData.segments || []).map(s => `[${formatTime(s.start)}] ${getDisplayName(s.speaker)}: ${s.text}`).join('\n');
            navigator.clipboard.writeText(text);
            showToast('Transcript copied to clipboard.', 'success');
        });

        downloadTxtBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            const text = (lastResponseData.segments || []).map(s =>
                `[${fmtStamp(s.start)} -> ${fmtStamp(s.end)}] ${getDisplayName(s.speaker)}: ${s.text}`
            ).join('\n');
            downloadFile(text, 'transcript.txt', 'text/plain');
        });

        downloadSrtBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            let srt = '';
            (lastResponseData.segments || []).forEach((s, idx) => {
                srt += `${idx + 1}\n${fmtStamp(s.start, true)} --> ${fmtStamp(s.end, true)}\n${getDisplayName(s.speaker)}: ${s.text}\n\n`;
            });
            downloadFile(srt, 'transcript.srt', 'text/plain');
        });

        downloadVttBtn.addEventListener('click', () => {
            if (!lastResponseData) return;
            let vtt = 'WEBVTT\n\n';
            (lastResponseData.segments || []).forEach((s) => {
                vtt += `${fmtStamp(s.start)} --> ${fmtStamp(s.end)}\n<v ${getDisplayName(s.speaker)}>${s.text}\n\n`;
            });
            downloadFile(vtt, 'transcript.vtt', 'text/vtt');
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


@app.get("/v1/progress/{job_id}")
def get_job_progress(job_id: str, _: bool = Depends(verify_api_key)):
    """Poll live pipeline progress for a submitted job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": {"message": "Unknown job id.", "code": "job_not_found"}}, status_code=404)
    return JSONResponse(job)


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
    asr_backend: str = Form("auto"),
    job_id: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key)
):
    """
    OpenAI-compatible speech transcription & speaker diarization endpoint.
    """
    MAX_UPLOAD_BYTES = 500 * 1024 * 1024
    file_suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(
                status_code=413,
                detail={"error": {"message": f"File too large ({len(content) / (1024*1024):.0f} MB) — limit is 500 MB.", "type": "invalid_request_error", "code": "file_too_large"}},
            )
        tmp.write(content)

    active_device = device or SERVER_CONFIG["default_device"]

    def _progress_cb(p):
        if job_id:
            entry = dict(p)
            entry["ts"] = time.time()
            with JOBS_LOCK:
                JOBS[job_id] = entry

    def _stash_result(payload):
        """Keep the final response under the job so a dropped client
        connection can recover it via /v1/progress polling."""
        if job_id:
            with JOBS_LOCK:
                entry = JOBS.get(job_id) or {}
                entry.update({"stage": "done", "detail": "Completed", "pct": 100.0,
                              "ts": time.time(), "result": payload})
                JOBS[job_id] = entry

    if job_id:
        now = time.time()
        with JOBS_LOCK:
            JOBS[job_id] = {"stage": "upload", "detail": "Received — queuing on server…", "pct": None, "ts": now}
            # Expire jobs older than 10 minutes and keep at most 32 done entries
            stale = [k for k, v in JOBS.items() if now - v.get("ts", 0) > 600]
            for k in stale:
                JOBS.pop(k, None)
            done = [k for k, v in JOBS.items() if v.get("stage") == "done"]
            for k in done[:-32]:
                JOBS.pop(k, None)

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
                padding_sec=0.15,
                asr_backend=asr_backend,
                progress_cb=_progress_cb
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
                        "avg_logprob": None,
                        "compression_ratio": None,
                        "no_speech_prob": None,
                        "npu_latency_ms": r.get("latency_ms", 0.0),
                        "backend": r.get("backend", "")
                    })

                response_payload = {
                    "task": "transcribe",
                    "language": language or "en",
                    "duration": round(duration, 2),
                    "text": full_text,
                    "segments": segments
                }
                _stash_result(response_payload)
                return JSONResponse(response_payload)

            else:  # standard json
                json_payload = {
                    "text": full_text,
                    "speakers_detected": len(set(r["speaker"] for r in results)),
                    "turns": results
                }
                _stash_result(json_payload)
                return JSONResponse(json_payload)

        else:
            text, _ = await run_pipeline_blocking(
                transcribe_audio_segment,
                audio_path=tmp_path,
                lemonade_url=SERVER_CONFIG["lemonade_url"],
                language=language,
                prompt=prompt,
                fallback_device=active_device,
                asr_backend=asr_backend
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

    except HTTPException:
        raise
    except RuntimeError as e:
        msg = str(e)
        if "Could not decode" in msg or "unsupported format" in msg.lower():
            raise HTTPException(
                status_code=422,
                detail={"error": {"message": msg, "type": "invalid_request_error", "code": "unsupported_audio_format"}},
            )
        raise HTTPException(
            status_code=500, detail={"error": {"message": msg, "type": "server_error"}}
        )
    except Exception as e:
        msg = str(e)
        if "Format not recognised" in msg or "Error opening" in msg:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "message": f"Could not decode audio file — unsupported or corrupt format. Try WAV, MP3, or FLAC. ({msg})",
                        "type": "invalid_request_error",
                        "code": "unsupported_audio_format",
                    }
                },
            )
        raise HTTPException(
            status_code=500, detail={"error": {"message": msg, "type": "server_error"}}
        )
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
        if api_key and api_key.strip().lower() == "empty":
            api_key = None  # vLLM-style sentinel meaning "no auth"
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
