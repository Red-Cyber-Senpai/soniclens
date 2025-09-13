# backend/diarization/diarize_and_transcribe.py
import json
import os
import subprocess
import tempfile
from typing import List, Dict

# try importing diarizer
try:
    from backend.diarization.diarize import simple_vad_segments
except Exception:
    simple_vad_segments = None

def run_diarize_via_script(path: str):
    # fallback: run existing script and parse printed python list
    cmd = ["python", "backend/diarization/diarize.py", path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout.strip()
    # naive parse: attempt to eval JSON-like output
    try:
        return json.loads(out)
    except Exception:
        # try eval (less safe) as a fallback
        import ast
        return ast.literal_eval(out)

def diarize_audio(path: str):
    if simple_vad_segments:
        return simple_vad_segments(path)
    else:
        return run_diarize_via_script(path)

def extract_clip(source: str, start: float, duration: float, dst: str):
    cmd = [
        "ffmpeg", "-y", "-i", source,
        "-ss", str(start), "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        dst
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_segments(source_wav: str, out_json: str, model_name="small", device=None):
    from backend.asr.asr_whisper import transcribe_file
    segs = diarize_audio(source_wav)
    out = []
    tmpdir = tempfile.mkdtemp(prefix="clip_")
    for i, s in enumerate(segs):
        start = max(0.0, s["start"] - 0.08)  # small pre-roll
        duration = (s["end"] - s["start"]) + 0.16
        clip = os.path.join(tmpdir, f"seg_{i}.wav")
        extract_clip(source_wav, start, duration, clip)
        try:
            txt = transcribe_file(clip, model_name=model_name, device=device)
        except Exception as e:
            txt = ""
            print("Transcription error for clip", i, e)
        out.append({
            "speaker": s.get("speaker", f"S{i+1}"),
            "start": s["start"],
            "end": s["end"],
            "text": txt
        })
    # cleanup clips
    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception:
        pass
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Wrote:", out_json)
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python diarize_and_transcribe.py input.wav output.json [--model MODEL] [--device cpu|mps|cuda]")
        sys.exit(1)
    src = sys.argv[1]
    dest = sys.argv[2]
    model = "small"
    device = None
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model")+1]
    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device")+1]
    transcribe_segments(src, dest, model_name=model, device=device)
