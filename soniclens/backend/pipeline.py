# backend/pipeline.py
"""
Chain: diarization -> cut segments (ffmpeg) -> transcribe each -> return list of speaker-tagged entries.
Usage:
    python backend/pipeline.py data/samples/meeting1.wav
"""
import os
import sys
import json
import subprocess
import tempfile

# -- Try to import helpers from your repo; fall back to script calls if import fails
try:
    from backend.diarization.diarize import simple_vad_segments
except Exception:
    simple_vad_segments = None

try:
    from backend.asr.asr_whisper import transcribe_file
except Exception:
    transcribe_file = None

def call_diarize_script(path):
    """Fallback: call diarize.py and parse its stdout (assumes it prints a Python list or JSON)."""
    cmd = [sys.executable, os.path.join("backend","diarization","diarize.py"), path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    # try parse JSON, else eval
    try:
        return json.loads(out)
    except Exception:
        try:
            return eval(out)
        except Exception:
            raise RuntimeError(f"Failed to parse diarize output: {out[:200]} (stderr: {res.stderr})")

def call_asr_script(path):
    """Fallback: call asr_whisper.py and return printed transcript (assumes it prints transcript)."""
    cmd = [sys.executable, os.path.join("backend","asr","asr_whisper.py"), path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    if out:
        return out
    if res.stderr:
        raise RuntimeError(f"ASR script stderr: {res.stderr}")
    return ""

def cut_audio_segment(src, start, end, out_path):
    """Cut with a small margin so words at edges are preserved."""
    s = max(0, start - 0.15)
    e = end + 0.15
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ss", str(s), "-to", str(e),
        "-ar", "16000", "-ac", "1", "-f", "wav", out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def diarize_and_transcribe(audio_path, language="en"):
    # 1) get segments
    if callable(simple_vad_segments):
        segments = simple_vad_segments(audio_path)
    else:
        segments = call_diarize_script(audio_path)

    # ensure segments is a list of dicts with start/end
    if not isinstance(segments, list):
        raise RuntimeError("diarize returned non-list")

    results = []
    for i, seg in enumerate(segments):
        # seg expected to have 'start' and 'end'; tolerate different keys
        start = seg.get("start") if isinstance(seg, dict) else seg[0]
        end = seg.get("end") if isinstance(seg, dict) else seg[1]
        speaker = seg.get("speaker", f"S{i%2+1}") if isinstance(seg, dict) else f"S{i%2+1}"
        # create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name
        try:
            cut_audio_segment(audio_path, float(start), float(end), seg_path)
        except Exception as e:
            # if cut fails, skip but record error
            results.append({"speaker": speaker, "start": start, "end": end, "text": f"[cut error: {e}]"})
            try: os.remove(seg_path)
            except: pass
            continue

        # transcribe using imported wrapper if available otherwise fallback
        try:
            if callable(transcribe_file):
                text = transcribe_file(seg_path, language=language)
            else:
                text = call_asr_script(seg_path)
        except Exception as e:
            text = f"[ASR error: {e}]"

        results.append({"speaker": speaker, "start": float(start), "end": float(end), "text": text})
        try: os.remove(seg_path)
        except: pass

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/pipeline.py <audio.wav>")
        sys.exit(1)
    audio = sys.argv[1]
    out = diarize_and_transcribe(audio)
    print(json.dumps(out, indent=2, ensure_ascii=False))
