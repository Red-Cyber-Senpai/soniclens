import re
try:
    from transformers import pipeline
except Exception:
    pipeline = None

SUM = None
def get_summarizer():
    global SUM
    if SUM is None and pipeline is not None:
        try:
            SUM = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            SUM = None
    return SUM

def extract_actions_from_text(text: str):
    if not text or len(text.strip()) < 10:
        return []
    summ = get_summarizer()
    if summ:
        try:
            summary = summ(text, max_length=80, min_length=10)[0]["summary_text"]
        except Exception:
            summary = text
    else:
        summary = text
    action_keywords = ["assign", "action", "decide", "should", "will", "deliver", "due"]
    sentences = re.split(r'(?<=[.!?]) +', summary)
    actions = [s.strip() for s in sentences if any(k in s.lower() for k in action_keywords)]
    if not actions:
        actions = [summary]
    return actions

if __name__ == "__main__":
    print(extract_actions_from_text("We will pilot at School X. Raj will prepare dataset. Action: create metrics by Friday."))
