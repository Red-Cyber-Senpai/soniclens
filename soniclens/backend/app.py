from fastapi import FastAPI, UploadFile, File
import shutil, os
import uvicorn
from typing import List

app = FastAPI(title="SonicLens Backend")

UPLOAD_DIR = "data/samples"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "SonicLens Backend Running 🚀"}

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    """
    Upload audio file -> run ASR (whisper wrapper) if available -> return transcript.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Try to run ASR if module exists; otherwise return placeholder string.
    try:
        from backend.asr.asr_whisper import transcribe_file
        text = transcribe_file(file_path)
    except Exception as e:
        text = f"Demo transcript placeholder. (ASR error: {e})"
    return {"filename": file.filename, "transcript": text}

@app.post("/diarize/")
async def diarize(file: UploadFile = File(...)):
    """
    Upload audio file -> run diarization (VAD+placeholder clustering) -> return segments.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from backend.diarization.diarize import simple_vad_segments
        segments = simple_vad_segments(file_path)
    except Exception as e:
        segments = {"error": f"diarization not available: {e}"}
    return {"filename": file.filename, "segments": segments}

@app.post("/actions/")
async def actions(body: dict):
    """
    Accept JSON body: {"text": "<transcript here>"} -> extracts action items.
    """
    text = body.get("text", "")
    try:
        from backend.summarizer.extract_actions import extract_actions_from_text
        actions = extract_actions_from_text(text)
    except Exception as e:
        actions = {"error": f"extractor not available: {e}"}
    return {"actions": actions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
