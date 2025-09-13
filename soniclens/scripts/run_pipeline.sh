#!/usr/bin/env bash
set -e
AUDIO=${1:-data/samples/meeting1.wav}
DEVICE=${WHISPER_DEVICE:-cpu}
export PYTHONPATH="$(pwd)"
python -m backend.diarization.diarize_and_transcribe "$AUDIO" diarized_with_text.json --model small --device "$DEVICE"
python backend/diarization/cleanup_transcript.py diarized_with_text.json diarized_clean.json
echo "Wrote diarized_with_text.json and diarized_clean.json"
