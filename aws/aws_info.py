# --------------------------------------------------------------
# aws_info.py  (VERSIONED vectorstores + ALWAYS-FRESH loading)
# --------------------------------------------------------------
# ✅ Fixes Windows file-lock problems (no deletes)
# ✅ Creates a NEW version when KB changes
# ✅ Always loads the LATEST version
# ✅ Compatible with your main.py + RAG_Agent.py
#
# Author: Ayush Garla
# --------------------------------------------------------------

import os
import re
import time
import json
import hashlib
import spacy

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --------------------------------------------------------------
# Paths / Globals
# --------------------------------------------------------------
AWS_KB_PATH = "aws/data/aws_knowledge_base.txt"
AWS_VECTORSTORE_VERSION_DIR = "aws/vectorstore_versions"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RELEVANCE_THRESHOLD = 1.35   # lower = more relevant

AWS_DB = None
AWS_RETRIEVER = None


# --------------------------------------------------------------
# Utils
# --------------------------------------------------------------
def _kb_exists():
    if not os.path.exists(AWS_KB_PATH):
        raise FileNotFoundError(f"❌ {AWS_KB_PATH} missing.")


def load_aws_kb() -> str:
    """Load KB text."""
    _kb_exists()
    with open(AWS_KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"📘 Loaded AWS KB ({len(text.split())} words)")
    return text


def clean_kb_text(raw: str) -> str:
    """Light KB cleanup."""
    raw = re.sub(r"[^a-zA-Z0-9.,!?'\s:/()_-]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def chunk_kb_text(text: str, max_words=400):
    """Sentence-aware chunking."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks, cur, wc = [], [], 0

    for sent in sentences:
        w = len(sent.split())
        if wc + w <= max_words:
            cur.append(sent)
            wc += w
        else:
            chunks.append(" ".join(cur))
            cur = [sent]
            wc = w

    if cur:
        chunks.append(" ".join(cur))

    print(f"🧩 Split KB into {len(chunks)} chunks")
    return chunks


def _kb_hash(text: str) -> str:
    """Stable hash so we know KB changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_meta(version_path: str, kb_hash: str):
    meta = {
        "kb_hash": kb_hash,
        "built_at": int(time.time()),
        "kb_path": AWS_KB_PATH
    }
    with open(os.path.join(version_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _read_meta(version_path: str):
    meta_path = os.path.join(version_path, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# --------------------------------------------------------------
# Build NEW vectorstore version
# --------------------------------------------------------------
def build_new_vectorstore():
    """
    Creates a NEW version folder:
      aws/vectorstore_versions/store_<timestamp>/
    """
    global AWS_DB, AWS_RETRIEVER

    os.makedirs(AWS_VECTORSTORE_VERSION_DIR, exist_ok=True)

    timestamp = int(time.time())
    version_path = os.path.join(AWS_VECTORSTORE_VERSION_DIR, f"store_{timestamp}")
    os.makedirs(version_path, exist_ok=True)

    raw = load_aws_kb()
    clean = clean_kb_text(raw)
    kb_hash = _kb_hash(clean)

    chunks = chunk_kb_text(clean)

    docs = [
        Document(page_content=c, metadata={"chunk_id": i + 1})
        for i, c in enumerate(chunks)
    ]

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    AWS_DB = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=version_path
    )
    AWS_RETRIEVER = AWS_DB.as_retriever(search_kwargs={"k": 3})

    _write_meta(version_path, kb_hash)

    print(f"✅ Built NEW AWS VectorStore: {version_path}")
    return version_path


# --------------------------------------------------------------
# Load newest vectorstore (auto-refresh if KB changed)
# --------------------------------------------------------------
def load_latest_vectorstore():
    """
    Loads the latest store version.
    If KB changed since last build -> builds a new store automatically.
    """
    global AWS_DB, AWS_RETRIEVER

    os.makedirs(AWS_VECTORSTORE_VERSION_DIR, exist_ok=True)

    # find newest version
    versions = sorted(os.listdir(AWS_VECTORSTORE_VERSION_DIR), reverse=True)
    latest_path = None
    if versions:
        latest_path = os.path.join(AWS_VECTORSTORE_VERSION_DIR, versions[0])

    # if none exist -> build first one
    if latest_path is None or not os.path.exists(latest_path):
        latest_path = build_new_vectorstore()

    # compare KB hash with latest meta
    raw = load_aws_kb()
    clean = clean_kb_text(raw)
    current_hash = _kb_hash(clean)

    meta = _read_meta(latest_path)
    last_hash = meta["kb_hash"] if meta and "kb_hash" in meta else None

    if last_hash != current_hash:
        print("♻️ KB changed since last build → creating new version...")
        latest_path = build_new_vectorstore()

    print(f"🔁 Loading AWS KB VectorStore from: {latest_path}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    AWS_DB = Chroma(
        persist_directory=latest_path,
        embedding_function=embeddings
    )
    AWS_RETRIEVER = AWS_DB.as_retriever(search_kwargs={"k": 3})
    return AWS_DB


# --------------------------------------------------------------
# Public accessor
# --------------------------------------------------------------
def get_aws_retriever():
    global AWS_RETRIEVER
    if AWS_RETRIEVER is None:
        load_latest_vectorstore()
    return AWS_RETRIEVER


# --------------------------------------------------------------
# Relevance check
# --------------------------------------------------------------
def is_topic_in_aws_kb(query: str) -> bool:
    global AWS_DB
    if AWS_DB is None:
        load_latest_vectorstore()

    results = AWS_DB.similarity_search_with_score(query, k=1)
    if not results:
        return False

    _, score = results[0]
    print(f"🔎 AWS KB relevance score: {score:.4f}")
    return score <= RELEVANCE_THRESHOLD


# --------------------------------------------------------------
# Append new learned content to KB
# --------------------------------------------------------------
def append_to_aws_kb(topic: str, explanation: str):
    """
    Append new KB block. Vectorstore is refreshed by calling rebuild later.
    """
    block = (
        "\n\n### New topic - self learning\n"
        f"#### Topic: {topic.strip()}\n\n"
        f"{explanation.strip()}\n"
    )

    with open(AWS_KB_PATH, "a", encoding="utf-8") as f:
        f.write(block)

    print("📘 Added new topic to AWS KB.")


# --------------------------------------------------------------
# Rebuild AWS KB vectorstore (NEW version)
# --------------------------------------------------------------
def rebuild_aws_kb_vectorstore():
    """
    Always creates a NEW version.
    No delete -> no lock issues.
    """
    print("🛠 Creating NEW AWS KB VectorStore version...")
    build_new_vectorstore()
    return True


# --------------------------------------------------------------
# Debug
# --------------------------------------------------------------
if __name__ == "__main__":
    load_latest_vectorstore()
