# SonicLens — MVP repo

Purpose: demo pipeline for AR + collaborative caption fusion + action-item extraction.

Quick start (Linux / macOS):
1. Clone repo
   git clone <your-repo-url>
   cd soniclens
2. Create virtualenv and activate
   python -m venv .venv && source .venv/bin/activate
3. Install deps
   pip install -r requirements.txt
4. Run backend server
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Example uses:
- Transcribe an audio file:
  python backend/asr/asr_whisper.py --input data/samples/test1.wav
- Run diarization:
  python backend/diarization/diarize.py --input data/samples/meeting1.wav

See `docs/` for more instructions and demo runbook.
