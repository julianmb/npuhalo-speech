# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0] - 2026-08-31

### Added
- Transcription Engine selector (`asr_backend`: `auto`/`npu`/`cpu`/`gpu`) across CLI (`--asr-backend`), API (`asr_backend` form field), and Web Studio.
- Live job progress: `GET /v1/progress/{job_id}` polling endpoint and progress stepper (upload % → diarize → per-turn transcription).
- Compressed audio support (M4A/AAC/WebM) via PyAV — no system `ffmpeg` required.
- Full-screen auth gate (API key validated once per device, stored in `localStorage`).
- Review workflow: playback speed (0.75×–2×), keyboard shortcuts (`Space`, `←/→`, `J/K/L`), follow-playback auto-scroll.
- Per-speaker talk-time percentages in the results summary.
- Inline search: dims non-matching turns, match counter, `Enter`/`Shift+Enter` navigation.
- Transcript history: last 5 results kept locally, restore after refresh.
- Connection-drop recovery: finished jobs recoverable after a network blip; retry button on failures.
- App-style viewport-fit desktop layout; duration preview and upload size guards (500 MB / 3 h).

### Fixed
- `librosa_or_sf_load` and diarization fallback now handle compressed formats via PyAV.
- Server returns `422` for undecodable audio and `413` for oversized uploads (was `500`).
- `transcribe_audio_segment` typed as `Tuple[str, str]`, thread-safe lazy Whisper load, cached NPU health check (prefers `/api/v1/health`), chronological turn ordering.
- `OPENAI_API_KEY=EMPTY` sentinel treated as no-auth.
- CORS `allow_credentials` removed; API key compared with `secrets.compare_digest`.

## [1.1.0] - 2026-08-31

- Initial public release: Whisper-v3-turbo on XDNA 2 NPU via Lemonade + pyannote diarization, OpenAI-compatible API, and browser studio.

[1.2.0]: https://github.com/julianmb/npuhalo-speech/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/julianmb/npuhalo-speech/releases/tag/v1.1.0
