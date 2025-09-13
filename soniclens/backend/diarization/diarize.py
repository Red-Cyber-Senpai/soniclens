import sys, contextlib, wave
import webrtcvad

def simple_vad_segments(path):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        if wf.getnchannels() != 1:
            raise Exception("Please provide mono WAV for demo diarization.")
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    vad = webrtcvad.Vad(3)
    frame_duration = 30  # ms
    bytes_per_frame = int(sample_rate * (frame_duration/1000.0) * 2)  # 16-bit
    segments = []
    inside = False
    start_time = 0.0
    for i in range(0, len(pcm), bytes_per_frame):
        chunk = pcm[i:i+bytes_per_frame]
        if len(chunk) < bytes_per_frame:
            break
        t = i / (sample_rate*2)
        is_speech = vad.is_speech(chunk, sample_rate)
        if is_speech and not inside:
            inside = True
            start_time = t
        elif not is_speech and inside:
            inside = False
            segments.append((start_time, t))
    if inside:
        segments.append((start_time, len(pcm)/(sample_rate*2)))
    results = []
    for i,(s,e) in enumerate(segments):
        results.append({"speaker": f"S{i%2+1}", "start": s, "end": e})
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diarize.py <audiofile>")
    else:
        print(simple_vad_segments(sys.argv[1]))
