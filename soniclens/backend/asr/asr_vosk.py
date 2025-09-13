#!/usr/bin/env python3
# backend/asr/asr_vosk.py
"""
Robust VOSK transcription helper.

Usage:
  python backend/asr/asr_vosk.py data/samples/test1.wav --model models/vosk-model-small-en-us-0.15

If input audio is not mono 16-bit WAV, the script will attempt to convert it (requires ffmpeg + pydub).
"""

import argparse
import json
import os
import sys
import tempfile
import wave
from vosk import Model, KaldiRecognizer

# Optional conversion
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except Exception:
    HAVE_PYDUB = False

CHUNK_BYTES = 4000  # typical chunk size used in VOSK examples

def is_mono_16bit_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnchannels() == 1 and wf.getsampwidth() == 2
    except wave.Error:
        return False

def convert_to_mono_16bit(src_path):
    """Convert any audio file to mono 16-bit WAV (16000 Hz) using pydub.
       Returns path to temporary WAV file (caller should remove it)."""
    if not HAVE_PYDUB:
        raise RuntimeError("pydub not installed. Install it with `pip install pydub` and ensure ffmpeg is on PATH.")
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio.export(tmp_path, format="wav")
    return tmp_path

def transcribe_vosk(wav_path, model_path, convert_if_needed=True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"VOSK model not found at '{model_path}'. Download and unzip a model into this path.")

    use_tmp = None
    wav_to_open = wav_path
    if not is_mono_16bit_wav(wav_path):
        if not convert_if_needed:
            raise ValueError("Input WAV must be mono 16-bit. Either convert the file or run with --convert.")
        if not HAVE_PYDUB:
            raise RuntimeError("Audio is not mono 16-bit and pydub/ffmpeg are not available to convert it.")
        print("[INFO] Converting input to mono 16-bit WAV (16000 Hz) using pydub/ffmpeg...")
        use_tmp = convert_to_mono_16bit(wav_path)
        wav_to_open = use_tmp

    # Open WAV and run recognizer
    with wave.open(wav_to_open, "rb") as wf:
        sample_rate = wf.getframerate()
        model = Model(model_path)
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)  # include word-level timing if available

        results = []
        while True:
            data = wf.readframes(CHUNK_BYTES)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                r = json.loads(rec.Result())
                text = r.get("text", "")
                if text:
                    results.append(text)
            else:
                # optional: handle partial results if you want streaming feedback
                # partial = json.loads(rec.PartialResult()).get("partial", "")
                # if partial: print("PARTIAL:", partial)
                pass

        final_r = json.loads(rec.FinalResult())
        final_text = final_r.get("text", "")
        if final_text:
            results.append(final_text)

    if use_tmp:
        try:
            os.remove(use_tmp)
        except Exception:
            pass

    return " ".join([s for s in results if s])

def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV using VOSK (mono 16-bit required).")
    parser.add_argument("wav", help="Path to WAV/audio file")
    parser.add_argument("--model", "-m", default="models/vosk-model-small-en-us-0.15", help="Path to VOSK model directory")
    parser.add_argument("--no-convert", action="store_true", help="Do not attempt to auto-convert non-mono WAV files")
    args = parser.parse_args()

    wav_path = args.wav
    model_path = args.model

    if not os.path.exists(wav_path):
        print(f"[ERROR] Audio file not found: {wav_path}", file=sys.stderr)
        sys.exit(2)

    try:
        transcript = transcribe_vosk(wav_path, model_path, convert_if_needed=not args.no_convert)
        print("=== TRANSCRIPT ===")
        print(transcript)
    except Exception as e:
        print("[ERROR]", str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
