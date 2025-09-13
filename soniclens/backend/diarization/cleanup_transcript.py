# backend/diarization/cleanup_transcript.py
import json
import re
import sys
from typing import List, Dict

# parameters you can tune
MIN_TEXT_LEN = 1            # minimum non-space chars to keep
MIN_SEGMENT_DURATION = 0.15 # seconds; drop segments shorter than this after merging
MAX_GAP_TO_MERGE = 0.5      # seconds; if gap between segments <= this and same speaker -> merge
MAX_SHORT_GAP_MERGE = 0.3   # merge across small gaps even if different speaker (optional)

def normalize_text(t: str) -> str:
    t = t.strip()
    if not t:
        return ""
    # replace weird whitespace
    t = re.sub(r"\s+", " ", t)
    # fix repeated punctuation
    t = re.sub(r"([,.!?]){2,}", r"\1", t)
    # basic capitalization: first char after sentence end
    t = t.lower()
    # naive sentence capitalization
    sentences = re.split(r'([.!?]\s*)', t)
    out = ""
    for i in range(0, len(sentences), 2):
        s = sentences[i].strip()
        trail = sentences[i+1] if i+1 < len(sentences) else ""
        if s:
            s = s[0].upper() + s[1:]
        out += s + trail
    out = out.strip()
    # ensure ending punctuation
    if out and out[-1] not in ".!?":
        out = out + "."
    return out

def merge_segments(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda x: x["start"])
    out = []
    cur = dict(segs[0])
    cur["text"] = cur.get("text","").strip()
    for s in segs[1:]:
        s_text = s.get("text","").strip()
        gap = s["start"] - cur["end"]
        # if same speaker and gap small => merge
        if s.get("speaker") == cur.get("speaker") and gap <= MAX_GAP_TO_MERGE:
            cur["end"] = max(cur["end"], s["end"])
            if s_text:
                if cur.get("text"):
                    cur["text"] = cur["text"].rstrip() + " " + s_text
                else:
                    cur["text"] = s_text
            continue
        # If different speaker but very small gap and both short => merge as well
        if gap <= MAX_SHORT_GAP_MERGE and (cur["end"]-cur["start"] < 1.0 or s["end"]-s["start"] < 1.0):
            # merge but prefer marking speaker as both (or keep the earlier speaker)
            cur["end"] = max(cur["end"], s["end"])
            if s_text:
                if cur.get("text"):
                    cur["text"] = cur["text"].rstrip() + " " + s_text
                else:
                    cur["text"] = s_text
            continue
        # otherwise flush current and start new
        out.append(cur)
        cur = dict(s)
        cur["text"] = cur.get("text","").strip()
    out.append(cur)
    # drop tiny segments and normalize text
    final = []
    for seg in out:
        dur = seg["end"] - seg["start"]
        seg["text"] = normalize_text(seg.get("text",""))
        if (not seg["text"] or len(seg["text"].strip()) < MIN_TEXT_LEN) and dur < MIN_SEGMENT_DURATION:
            # drop
            continue
        final.append(seg)
    return final

def main():
    if len(sys.argv) < 2:
        print("Usage: python cleanup_transcript.py input.json [output.json]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) >=3 else None
    with open(inp,"r",encoding="utf-8") as f:
        data = json.load(f)
    cleaned = merge_segments(data)
    if outp:
        with open(outp,"w",encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print("Wrote:", outp)
    else:
        print(json.dumps(cleaned, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
