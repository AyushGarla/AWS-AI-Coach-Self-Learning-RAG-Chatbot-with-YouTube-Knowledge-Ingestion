# --------------------------------------------------------------
# NLP.py   (FINAL — clean, stable, chunk-safe version)
# --------------------------------------------------------------
# This module:
# 1. Loads transcript.txt
# 2. Cleans unwanted noise
# 3. Splits into clean text chunks using spaCy
# 4. Saves:
#       - clean_transcript.txt
#       - chunks.txt   (strict format for vector_store.py)
#
# 100% compatible with versioned AWS KB + transcript vectorstore.
# --------------------------------------------------------------

import os
import re
import spacy

# --------------------------------------------------------------
# Load transcript.txt
# --------------------------------------------------------------
def load_transcript(path="transcript.txt"):
    """Load transcript file and return raw text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {path} not found. Run transcript_extraction.py first.")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"✅ Loaded transcript ({len(text.split())} words).")
    return text


# --------------------------------------------------------------
# Clean transcript text
# --------------------------------------------------------------
def clean_transcript(raw_text):
    """
    Strong cleaning:
    - Remove timestamps (00:12, 01:23:44)
    - Remove bracketed noise: [Music], (Applause)
    - Remove weird symbols
    - Normalize whitespace
    """

    # remove timestamps like 00:10, 12:30:09
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", raw_text)

    # remove bracketed text
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)

    # remove any non-text junk
    text = re.sub(r"[^A-Za-z0-9.,!?'\s]", " ", text)

    # normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    # lowercase for consistency
    text = text.lower()

    print("🧹 Cleaned transcript text.")
    return text


# --------------------------------------------------------------
# Chunk transcript into semantic blocks
# --------------------------------------------------------------
def chunk_text(clean_text, max_words=500, output_path="chunks.txt"):
    """
    Creates a strictly formatted chunks.txt file:
        ### CHUNK 1
        text text...

        ### CHUNK 2
        ...
    """

    print("🧠 Splitting transcript into chunks...")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(clean_text)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = []
    word_count = 0

    for sent in sentences:
        w = len(sent.split())

        if word_count + w <= max_words:
            current_chunk.append(sent)
            word_count += w
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            word_count = w

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Save chunks.txt in strict format
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"### CHUNK {i}\n")
            f.write(chunk + "\n\n")

    print(f"✅ Created {len(chunks)} chunks and saved to '{output_path}'.")
    return chunks


# --------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------
def run_nlp_pipeline():
    """Runs all preprocessing steps in correct order."""
    raw_text = load_transcript()
    clean_text = clean_transcript(raw_text)

    # Save cleaned transcript
    with open("clean_transcript.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)
    print("✅ Saved 'clean_transcript.txt'.")

    chunks = chunk_text(clean_text)

    print("🎯 NLP preprocessing complete.")
    return chunks


# --------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    run_nlp_pipeline()
