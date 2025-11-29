# --------------------------------------------------------------
# kb_chunker.py  (Markdown-aware AWS Knowledge Base Chunker)
# FIXED VERSION — handles dict chunks & dedupes correctly
# --------------------------------------------------------------

import re
import os
from typing import List, Dict

KB_PATH = "aws/data/aws_knowledge_base.txt"

# --------- Utility -------------------------------------------------

def load_kb_text():
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"❌ Cannot find {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Remove repeated blank lines and normalize spacing."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------- Markdown Parsing ----------------------------------------

def extract_sections(text: str) -> Dict[str, str]:
    """
    Splits the KB into sections by markdown heading level 2:
        ## Amazon S3
        ## Amazon EC2
        ## VPC
        ## IAM
    Returns dict: { "Amazon S3": "...section text..." }
    """

    pattern = r"(##\s+[^\n]+)"
    parts = re.split(pattern, text)
    sections = {}

    # parts looks like [pretext, "## Heading1", text1, "## Heading2", text2 ...]
    i = 1
    while i < len(parts):
        heading = parts[i].replace("##", "").strip()
        body = parts[i + 1]
        sections[heading] = clean_text(body)
        i += 2

    return sections


# --------- Chunking logic ------------------------------------------

def chunk_large_section(text: str, max_words=180) -> List[str]:
    """
    Splits long sections into smaller sub-chunks of ~180 words.
    Splits ONLY on paragraph boundaries (blank line).
    """

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    wc = 0

    for p in paragraphs:
        words = len(p.split())
        if wc + words > max_words:
            chunks.append("\n\n".join(current))
            current = [p]
            wc = words
        else:
            current.append(p)
            wc += words

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove duplicate chunks.  
    Deduplication key = FIRST 200 chars of 'text'.
    """

    seen = set()
    clean = []

    for c in chunks:
        # Ensure chunk is a dict
        if not isinstance(c, dict) or "text" not in c:
            continue

        text = str(c["text"])
        key = text[:200]   # safe, always a string

        if key not in seen:
            clean.append(c)
            seen.add(key)

    return clean


# --------- MAIN CHUNK PIPELINE -------------------------------------

def build_kb_chunks() -> List[Dict]:
    """
    Returns:
        [
            {
                "service": "Amazon S3",
                "chunk_id": "Amazon S3 — 1",
                "text": "... chunk text ..."
            },
            ...
        ]
    """
    raw = load_kb_text()
    sections = extract_sections(raw)

    final_chunks = []

    for service, body in sections.items():
        words = len(body.split())

        # split into sub-chunks if big
        if words > 220:
            subchunks = chunk_large_section(body)
        else:
            subchunks = [body]

        # store with metadata
        for i, ch in enumerate(subchunks, 1):
            final_chunks.append({
                "service": service,
                "chunk_id": f"{service} — {i}",
                "text": ch.strip()
            })

    return dedupe_chunks(final_chunks)
