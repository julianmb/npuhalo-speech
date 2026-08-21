#!/usr/bin/env python3
"""
transcribe_diarize.py — Heterogeneous Speech Pipeline (Cross-Platform: Linux, Windows, Docker)
- Transcription: OpenAI Whisper-v3-turbo on AMD XDNA 2 NPU (/dev/accel/accel0 or Lemonade API),
                 with automatic graceful fallback to CPU/GPU Faster-Whisper.
- Diarization:   pyannote/speaker-diarization-community-1 on CPU or GPU (ROCm/CUDA).
- Features:      150ms acoustic boundary padding, prompt context memory, multi-file/batch processing.
- Output:        Speaker-attributed transcript (Text, SRT, VTT, JSON).
"""

import os
import sys
import glob
import json
import time
import argparse
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import requests
import numpy as np
import soundfile as sf
import torch

# Global cached local fallback model
_LOCAL_WHISPER_MODEL = None
_LOCAL_WHISPER_LOCK = threading.Lock()

# Cached NPU availability (avoids a health-check round trip per segment)
_NPU_STATUS = {"url": None, "available": False, "checked_at": 0.0}
_NPU_CACHE_TTL = 30.0


def color(text: str, code: str) -> str: return f"\033[{code}m{text}\033[0m"
def green(text: str) -> str: return color(text, "1;32")
def yellow(text: str) -> str: return color(text, "1;33")
def cyan(text: str) -> str: return color(text, "1;36")
def red(text: str) -> str: return color(text, "1;31")
def bold(text: str) -> str: return color(text, "1")
def dim(text: str) -> str: return color(text, "2")


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Format seconds into HH:MM:SS.mmm or HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if srt_format:
        return f"{hrs:02d}:{mins:02d}:{int(secs):02d},{int((secs % 1) * 1000):03d}"
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Retrieve Hugging Face token from explicit arg, env var, or cached token file."""
    if explicit_token:
        return explicit_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        try:
            return token_path.read_text().strip()
        except Exception:
            pass
    return None


def resolve_torch_device(requested_device: str = "cpu") -> str:
    """Validate requested device and fall back gracefully if unavailable."""
    req = requested_device.lower().strip()
    if req in ("cuda", "rocm", "gpu", "auto"):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "ROCm/CUDA GPU"
            print(green(f"  • Diarization Acceleration: GPU detected ({gpu_name})"))
            return "cuda"
        else:
            if req != "auto":
                print(yellow(f"  [!] GPU/ROCm requested ('{requested_device}') but PyTorch CUDA/ROCm backend not found. Falling back to CPU."))
            return "cpu"
    return "cpu"


def check_npu_available(lemonade_url: str = "http://127.0.0.1:13305") -> bool:
    """Verify if Lemonade NPU daemon is reachable (cached for _NPU_CACHE_TTL seconds)."""
    now = time.time()
    if (
        _NPU_STATUS["url"] == lemonade_url
        and now - _NPU_STATUS["checked_at"] < _NPU_CACHE_TTL
    ):
        return _NPU_STATUS["available"]
    available = False
    try:
        res = requests.get(f"{lemonade_url}/health", timeout=1.5)
        available = res.status_code == 200
    except Exception:
        available = False
    _NPU_STATUS.update({"url": lemonade_url, "available": available, "checked_at": now})
    return available


def ensure_npu_whisper(lemonade_url: str = "http://127.0.0.1:13305") -> bool:
    return check_npu_available(lemonade_url)


def get_local_whisper_model(device: str = "cpu", model_size: str = "turbo"):
    """Lazy load faster-whisper on CPU or CUDA (thread-safe)."""
    global _LOCAL_WHISPER_MODEL
    if _LOCAL_WHISPER_MODEL is None:
        with _LOCAL_WHISPER_LOCK:
            if _LOCAL_WHISPER_MODEL is None:
                from faster_whisper import WhisperModel
                compute_type = "float16" if (device == "cuda" and torch.cuda.is_available()) else "int8"
                target_dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
                print(cyan(f"[*] Loading local Faster-Whisper ({model_size}) on {target_dev.upper()} ({compute_type})..."))
                _LOCAL_WHISPER_MODEL = WhisperModel(model_size, device=target_dev, compute_type=compute_type)
    return _LOCAL_WHISPER_MODEL


def transcribe_audio_segment(
    audio_path: str,
    lemonade_url: str = "http://127.0.0.1:13305",
    model_name: str = "whisper-v3-turbo-FLM",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    fallback_device: str = "cpu"
) -> Tuple[str, str]:
    """
    Transcribe audio segment with prompt conditioning.
    Attempts NPU first via Lemonade; falls back to local CPU/GPU Faster-Whisper if NPU is offline.
    """
    # 1. Try NPU via Lemonade
    if check_npu_available(lemonade_url):
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/wav")}
                data = {"model": model_name, "response_format": "json"}
                if language:
                    data["language"] = language
                if prompt:
                    data["prompt"] = prompt
                endpoint = f"{lemonade_url}/v1/audio/transcriptions"
                res = requests.post(endpoint, files=files, data=data, timeout=60)
                if res.status_code == 200:
                    return res.json().get("text", "").strip(), "NPU (XDNA 2)"
        except Exception:
            pass

    # 2. Local Fallback (CPU / GPU)
    model = get_local_whisper_model(device=fallback_device, model_size="turbo")
    segments, _ = model.transcribe(audio_path, language=language, initial_prompt=prompt, beam_size=1)
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    backend_label = f"Local Whisper ({fallback_device.upper()})"
    return text, backend_label


def transcribe_audio_npu(
    audio_path: str,
    lemonade_url: str = "http://127.0.0.1:13305",
    model_name: str = "whisper-v3-turbo-FLM",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    fallback_device: str = "cpu"
) -> str:
    text, _ = transcribe_audio_segment(audio_path, lemonade_url, model_name, language, prompt, fallback_device)
    return text


def run_pyannote_diarization(
    audio_path: str,
    hf_token: Optional[str],
    device: str = "cpu",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Run pyannote/speaker-diarization-community-1 on CPU or GPU.
    """
    from pyannote.audio import Pipeline

    token = get_hf_token(hf_token)
    model_id = "pyannote/speaker-diarization-community-1"
    target_device = torch.device(device)
    
    device_label = "CPU" if device == "cpu" else f"GPU ({device})"
    print(cyan(f"[*] Initializing {model_id} on {device_label}..."))
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
        pipeline.to(target_device)
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "gated" in err_msg or "NoneType" in err_msg:
            print(yellow("\n[!] Pyannote Community-1 is gated on Hugging Face (or no token provided)."))
            print(yellow(f"    Falling back to Acoustic VAD & Clustering on {device_label} for this run.\n"))
            return run_fallback_diarization(audio_path, num_speakers=num_speakers)
        raise e

    waveform, sample_rate = sf.read(audio_path)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    
    audio_tensor = torch.from_numpy(waveform.astype(np.float32)).to(target_device)
    audio_input = {"waveform": audio_tensor, "sample_rate": sample_rate}

    diarization_params = {}
    if num_speakers is not None:
        diarization_params["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarization_params["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarization_params["max_speakers"] = max_speakers

    print(cyan(f"[*] Processing diarization turns on {device_label}..."))
    t0 = time.time()
    diarization_result = pipeline(audio_input, **diarization_params)
    elapsed = time.time() - t0
    print(green(f"[+] Diarization on {device_label} completed in {elapsed:.2f}s"))

    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })

    return segments


def run_fallback_diarization(audio_path: str, num_speakers: Optional[int] = 2) -> List[Dict[str, Any]]:
    """Fallback voice activity & spectral clustering diarizer."""
    import librosa
    from sklearn.cluster import KMeans

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    
    intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]

    features = []
    valid_intervals = []
    
    for start_idx, end_idx in intervals:
        seg_dur = (end_idx - start_idx) / sr
        if seg_dur < 0.4:
            continue
        seg_y = y[start_idx:end_idx]
        mfcc = librosa.feature.mfcc(y=seg_y, sr=sr, n_mfcc=13)
        feat = np.mean(mfcc, axis=1)
        features.append(feat)
        valid_intervals.append((start_idx / sr, end_idx / sr))

    if len(features) == 0:
        return [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]

    k = min(num_speakers or 2, len(features))
    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(np.array(features))
    else:
        labels = [0] * len(features)

    segments = []
    for (st, en), label in zip(valid_intervals, labels):
        segments.append({
            "start": round(st, 3),
            "end": round(en, 3),
            "speaker": f"SPEAKER_{label:02d}"
        })

    merged = []
    for s in segments:
        if not merged:
            merged.append(s)
        else:
            prev = merged[-1]
            if prev["speaker"] == s["speaker"] and (s["start"] - prev["end"]) < 0.6:
                prev["end"] = s["end"]
            else:
                merged.append(s)

    return merged


def process_pipeline(
    audio_path: str,
    device: str = "cpu",
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    lemonade_url: str = "http://127.0.0.1:13305",
    language: Optional[str] = None,
    padding_sec: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Heterogeneous orchestration with 150ms acoustic padding & rolling prompt memory:
    1. Resample / load audio
    2. Speaker Diarization on CPU or GPU (Pyannote)
    3. Whisper transcription with boundary padding & context memory
    """
    audio_file = Path(audio_path).resolve()
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    active_device = resolve_torch_device(device)
    device_name = "CPU" if active_device == "cpu" else f"GPU ({active_device})"

    has_npu = check_npu_available(lemonade_url)
    asr_desc = f"{green('Whisper-v3-turbo on AMD XDNA 2 NPU')} ({lemonade_url})" if has_npu else f"{yellow('Local Faster-Whisper')} ({device_name})"

    print("\n" + "=" * 78)
    print(bold(" 🎙️  HETEROGENEOUS SPEECH PIPELINE (ASR + SPEAKER DIARIZATION)"))
    print("=" * 78)
    print(f"  • Input Audio:      {cyan(str(audio_file))}")
    print(f"  • ASR Engine:       {asr_desc}")
    print(f"  • Diarization:      {cyan('pyannote')} on {cyan(device_name)}")
    print(f"  • Acoustic Padding: {green(f'{int(padding_sec*1000)}ms')} per boundary (prevents clipped words)")
    print("-" * 78)

    # 1. Load Audio and convert to 16kHz mono WAV if needed
    y, sr = librosa_or_sf_load(str(audio_file))
    duration = len(y) / sr
    print(f"  • Audio Duration:   {duration:.2f} seconds ({sr} Hz)")

    # 2. Run Speaker Diarization
    speaker_turns = run_pyannote_diarization(
        str(audio_file),
        hf_token=hf_token,
        device=active_device,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    if not speaker_turns:
        speaker_turns = [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]

    # Ensure chronological order regardless of diarizer output ordering
    speaker_turns.sort(key=lambda t: (t["start"], t["end"]))

    print(f"[+] Found {len(speaker_turns)} speaker speech turns.")

    # 3. Transcribe each turn with padding & prompt context memory
    print(cyan("\n[*] Transcribing speaker turns with acoustic padding..."))
    results = []
    rolling_prompt = ""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, turn in enumerate(speaker_turns, start=1):
            st, en, spk = turn["start"], turn["end"], turn["speaker"]
            seg_len = en - st
            if seg_len < 0.2:
                continue

            # Apply 150ms acoustic padding at boundaries
            pad_start = max(0.0, st - padding_sec)
            pad_end = min(duration, en + padding_sec)
            
            start_sample = int(pad_start * sr)
            end_sample = min(len(y), int(pad_end * sr))
            segment_audio = y[start_sample:end_sample]

            temp_chunk_path = Path(tmpdir) / f"turn_{idx:03d}_{spk}.wav"
            sf.write(str(temp_chunk_path), segment_audio, sr)

            t0 = time.time()
            text, backend = transcribe_audio_segment(
                str(temp_chunk_path),
                lemonade_url=lemonade_url,
                language=language,
                prompt=rolling_prompt[-150:] if rolling_prompt else None,
                fallback_device=active_device
            )
            t_ms = (time.time() - t0) * 1000

            if text.strip():
                rolling_prompt += " " + text.strip()
                results.append({
                    "start": st,
                    "end": en,
                    "speaker": spk,
                    "text": text,
                    "latency_ms": round(t_ms, 1),
                    "backend": backend
                })
                print(f"  [{format_timestamp(st)} -> {format_timestamp(en)}] {bold(spk)}: {text} {dim(f'({t_ms:.0f}ms {backend})')}")

    return results


def librosa_or_sf_load(path: str):
    """Load audio, resample to 16kHz mono if necessary."""
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        return data.astype(np.float32), sr
    except Exception:
        import librosa
        data, sr = librosa.load(path, sr=16000)
        return data.astype(np.float32), sr


def output_transcript(results: List[Dict[str, Any]], out_format: str = "text", output_file: Optional[str] = None):
    """Format and output the aligned transcript."""
    lines = []
    if out_format == "json":
        content = json.dumps(results, indent=2, ensure_ascii=False)
    elif out_format in ("srt", "vtt"):
        is_vtt = out_format == "vtt"
        if is_vtt:
            lines.append("WEBVTT\n")
        for i, r in enumerate(results, start=1):
            st = format_timestamp(r["start"], srt_format=not is_vtt)
            en = format_timestamp(r["end"], srt_format=not is_vtt)
            if is_vtt:
                lines.append(f"{st} --> {en}\n<v {r['speaker']}>{r['text']}\n")
            else:
                lines.append(f"{i}\n{st} --> {en}\n{r['speaker']}: {r['text']}\n")
        content = "\n".join(lines)
    else:  # standard text
        for r in results:
            st = format_timestamp(r["start"])
            en = format_timestamp(r["end"])
            lines.append(f"[{st} -> {en}] {r['speaker']}: {r['text']}")
        content = "\n".join(lines)

    if output_file:
        Path(output_file).write_text(content, encoding="utf-8")
        print(green(f"\n[✔] Transcript saved to: {output_file}"))

    return content


def discover_audio_files(inputs: List[str]) -> List[Path]:
    """Expand file paths, wildcards, and directories into a unique list of audio files."""
    valid_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
    discovered = []

    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in valid_exts:
                    discovered.append(f)
        elif "*" in inp or "?" in inp:
            for match in glob.glob(inp, recursive=True):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() in valid_exts:
                    discovered.append(mp)
        elif p.is_file():
            discovered.append(p)

    return sorted(list(set(discovered)))


def main():
    parser = argparse.ArgumentParser(
        description="Heterogeneous Audio Pipeline: Whisper on NPU/CPU/GPU + Pyannote Diarization (Single File or Batch Mode)"
    )
    parser.add_argument("audio", nargs="*", help="Audio file(s), wildcards, or directories to process in batch")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda", "rocm", "gpu", "auto"], default="cpu",
                        help="Execution device for speaker diarization / fallback ASR (default: cpu)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers if known")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers (if exact count unknown)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers (if exact count unknown)")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g. 'en', 'es', 'zh')")
    parser.add_argument("--format", choices=["text", "json", "srt", "vtt"], default="text", help="Output format")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save transcript to output file (single file mode)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for batch processing")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for gated pyannote model")
    parser.add_argument("--lemonade-url", type=str, default=os.environ.get("LEMONADE_URL", "http://127.0.0.1:13305"), help="Lemonade NPU API URL")

    args = parser.parse_args()

    if not args.audio:
        parser.print_help()
        sys.exit(1)

    files = discover_audio_files(args.audio)
    if not files:
        print(red(f"[Error] No supported audio files found matching: {args.audio}"))
        sys.exit(1)

    is_batch = len(files) > 1 or args.output_dir is not None

    if is_batch:
        out_dir = Path(args.output_dir) if args.output_dir else Path("./transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(bold(f"\n📦 BATCH PROCESSING: {len(files)} audio files detected. Output directory: {out_dir}\n"))

        for idx, file_path in enumerate(files, start=1):
            print("=" * 78)
            print(bold(f" [{idx}/{len(files)}] Processing: {file_path.name}"))
            print("=" * 78)
            
            ext_map = {"text": ".txt", "json": ".json", "srt": ".srt", "vtt": ".vtt"}
            out_file = out_dir / f"{file_path.stem}{ext_map[args.format]}"
            
            try:
                results = process_pipeline(
                    audio_path=str(file_path),
                    device=args.device,
                    hf_token=args.hf_token,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                    lemonade_url=args.lemonade_url,
                    language=args.language
                )
                output_transcript(results, out_format=args.format, output_file=str(out_file))
            except Exception as e:
                print(red(f"[!] Failed processing {file_path.name}: {e}"))
        
        print(green(f"\n[✔] Batch processing finished! All transcripts saved to: {out_dir}\n"))

    else:
        # Single file execution
        file_path = files[0]
        try:
            results = process_pipeline(
                audio_path=str(file_path),
                device=args.device,
                hf_token=args.hf_token,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                lemonade_url=args.lemonade_url,
                language=args.language
            )
            print("\n" + "=" * 78)
            print(bold(" 📝 FINAL SPEAKER-ATTRIBUTED TRANSCRIPT"))
            print("=" * 78)
            content = output_transcript(results, out_format=args.format, output_file=args.output)
            if not args.output:
                print(content)
            print("=" * 78 + "\n")
        except Exception as e:
            print(red(f"\n[Error] Pipeline execution failed: {e}"))
            sys.exit(1)


if __name__ == "__main__":
    main()
