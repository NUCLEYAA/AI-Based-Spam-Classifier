import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\b")

def clean_text(text):
    """Normalize message text while preserving useful spam clues."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
