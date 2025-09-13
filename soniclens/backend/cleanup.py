# backend/cleanup.py
"""
Cleans raw pipeline JSON:
- removes empty text segments
- merges adjacent segments from same speaker with small gaps
- basic text normalization (remove weird chars, collapse whitespace, capitalize)
Usage:
    python backend/cleanup.py raw.json > polished.json
"""
import sys, json, re

def clean_text(s):
    if not s:
        return ""
    # remove non-UTF-8-like garbage, keep basic punctuation
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([?.!,])", r"\1", s)
    if s:
        s = s[0].upper() + s[1:]
    return s

def merge_segments(segments, max_gap=0.5):
    # segments: list of dicts sorted by start
    segs = sorted(segments, key=lambda x: x.get("start",0))
    # clean text
    for s in segs:
        s["text"] = clean_text(s.get("text",""))
    # drop empties
    segs = [s for s in segs if s["text"].strip() != ""]
    if not segs:
        print("[]")
        return
    out = [segs[0].copy()]
    for s in segs[1:]:
        cur = out[-1]
        same_speaker = (s.get("speaker") == cur.get("speaker"))
        gap = s.get("start",0) - cur.get("end",0)
        if same_speaker and gap <= max_gap:
            # merge
            cur["end"] = s["end"]
            cur["text"] = (cur["text"] + " " + s["text"]).strip()
        else:
            out.append(s.copy())
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/cleanup.py raw.json")
        sys.exit(1)
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    merge_segments(data)
