# backend/live_pipeline.py
# Dependencies: sounddevice, soundfile, webrtcvad, whisper (or openai-whisper)
# Usage: python backend/live_pipeline.py --device <device_index> --model tiny

import argparse
import tempfile
import time
import os
import queue
import subprocess
import json
import sounddevice as sd
import soundfile as sf
import webrtcvad
from backend.asr.asr_whisper import transcribe_file      # uses your existing wrapper
from backend.diarization.diarize import simple_vad_segments  # optional

# CONFIG
CHUNK_SEC = 6            # length of each chunk (s)
OVERLAP_SEC = 1          # overlap between chunks (s)
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_MODE = 2             # 0..3 (aggressiveness)

out_folder = "data/live_chunks"
os.makedirs(out_folder, exist_ok=True)

def record_chunk(device=None, seconds=CHUNK_SEC, sr=SAMPLE_RATE):
    data = sd.rec(int(seconds * sr), samplerate=sr, channels=CHANNELS, dtype='int16', device=device)
    sd.wait()
    return data

def write_wav(path, data, sr=SAMPLE_RATE):
    sf.write(path, data, sr, subtype='PCM_16')

def vad_has_voice(wav_path):
    # simple: load bytes -> run webrtcvad on frames
    import wave
    wf = wave.open(wav_path, 'rb')
    vad = webrtcvad.Vad(VAD_MODE)
    sample_rate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    # split into 30ms frames required by webrtcvad
    frame_duration_ms = 30
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit -> 2 bytes per sample
    voiced = False
    for i in range(0, len(frames), n):
        chunk = frames[i:i+n]
        if len(chunk) < n:
            break
        if vad.is_speech(chunk, sample_rate):
            voiced = True
            break
    wf.close()
    return voiced

def run_live(device=None, model="small", language="en"):
    print("Live pipeline starting — press Ctrl+C to stop")
    ring = queue.Queue()
    try:
        # rolling buffer approach: write overlapping chunks and process
        while True:
            t0 = time.time()
            # record chunk
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=out_folder)
            tmp_name = tmp.name
            tmp.close()
            data = record_chunk(device=device, seconds=CHUNK_SEC)
            write_wav(tmp_name, data, SAMPLE_RATE)
            # run VAD
            has_voice = vad_has_voice(tmp_name)
            print(f"[{time.strftime('%H:%M:%S')}] Recorded {tmp_name} — voice:{has_voice}")
            if has_voice:
                try:
                    text = transcribe_file(tmp_name, language=language)
                except Exception as e:
                    text = f"[ASR error: {e}]"
                now = time.time()
                seg = {
                    "filename": os.path.basename(tmp_name),
                    "start_ts": now - CHUNK_SEC,
                    "end_ts": now,
                    "text": text
                }
                print("TRANSCRIPT:", text)
                ring.put(seg)
            else:
                # optional: keep non-voice for context
                pass

            # remove old temp files so disk doesn't fill
            for f in os.listdir(out_folder):
                path = os.path.join(out_folder, f)
                if time.time() - os.path.getmtime(path) > 300:
                    try:
                        os.remove(path)
                    except:
                        pass

            # sleep only to create the desired overlap
            elapsed = time.time() - t0
            sleep_time = max(0.01, CHUNK_SEC - elapsed - OVERLAP_SEC)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping live capture.")
        # collect ring contents and print JSON
        items = []
        while not ring.empty():
            items.append(ring.get())
        print(json.dumps(items, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model", default="small", help="whisper model to use (tiny, small, medium, large-v3)")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    run_live(device=args.device, model=args.model, language=args.language)
