# backend/asr/asr_whisper.py
import os
from typing import Optional
import whisper

_MODEL_CACHE = {}

def _preferred_device(device: Optional[str]) -> str:
    # priority: explicit arg > WHISPER_DEVICE env var > cpu
    if device:
        return device
    return os.environ.get("WHISPER_DEVICE", "cpu")

def load_model(name: str = "small", device: Optional[str] = None):
    device = _preferred_device(device)
    cache_key = f"{name}:{device}"
    if cache_key not in _MODEL_CACHE:
        # whisper.load_model accepts a device parameter
        # (if your whisper version doesn't support it, you can set torch.device beforehand)
        _MODEL_CACHE[cache_key] = whisper.load_model(name, device=device)
    return _MODEL_CACHE[cache_key]

def transcribe_file(path: str, language: str = "en", model_name: str = "small", device: Optional[str] = None) -> str:
    """
    Transcribe a single file and return the text.
    Use model_name and device to control speed/accuracy and hardware.
    """
    model = load_model(model_name, device=device)
    # tune these parameters as needed:
    res = model.transcribe(path,
                           language=language,
                           condition_on_previous_text=False,
                           no_speech_threshold=0.6,
                           logprob_threshold=-1.0,
                           compression_ratio_threshold=2.4)
    return res.get("text", "").strip()

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="path to audio file")
    p.add_argument("--model", default="small", help="whisper model name")
    p.add_argument("--device", default=None, help="device: cpu / mps / cuda")
    p.add_argument("--language", default="en")
    args = p.parse_args()
    print(transcribe_file(args.audio, language=args.language, model_name=args.model, device=args.device))
