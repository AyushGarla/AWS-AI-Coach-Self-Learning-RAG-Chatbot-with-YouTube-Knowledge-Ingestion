# --------------------------------------------------------------
# vector_store.py (FINAL — supports AWS KB + Transcript vectorstores)
# --------------------------------------------------------------
# Handles:
#   1. Building AWS KB vectorstore from metadata-rich chunks
#   2. Building Transcript vectorstore from chunks.txt
#   3. Loading vectorstores safely for RAG agent
#
# Compatible with:
# - kb_chunker.py
# - RAG_Agent.py
# - aws_info.py
# --------------------------------------------------------------

import os
import glob
import shutil
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Directory paths
TRANSCRIPT_CHUNKS_FILE = "chunks.txt"
TRANSCRIPT_STORE_DIR = "vectorstore"
AWS_STORE_BASE = "aws/vectorstore_versions"


# =====================================================================
# 1. LOAD TRANSCRIPT CHUNKS
# =====================================================================

def load_transcript_chunks(path=TRANSCRIPT_CHUNKS_FILE):
    """Reads chunks.txt → returns list of text chunks."""
    if not os.path.exists(path):
        print(f"❌ Missing {path}. Run NLP.py first.")
        return []

    chunks = []
    current_chunk = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # new chunk starts
            if line.startswith("### CHUNK"):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = []
            else:
                if line:
                    current_chunk.append(line)

        # last chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

    print(f"📄 Loaded {len(chunks)} transcript chunks.")
    return chunks


# =====================================================================
# 2. BUILD TRANSCRIPT VECTORSTORE
# =====================================================================

def build_transcript_vectorstore():
    """Rebuild transcript vectorstore."""
    print("🛠 Rebuilding transcript vectorstore...")

    if os.path.exists(TRANSCRIPT_STORE_DIR):
        shutil.rmtree(TRANSCRIPT_STORE_DIR)

    chunks = load_transcript_chunks()
    if not chunks:
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        collection_name="transcript_store",
        persist_directory=TRANSCRIPT_STORE_DIR,
        embedding_function=embeddings
    )

    docs = [
        Document(
            page_content=c,
            metadata={"source": "transcript"}
        ) for c in chunks
    ]

    db.add_documents(docs)

    print("✅ Transcript vectorstore built.")
    return db


def load_transcript_vectorstore():
    """Load or rebuild transcript vectorstore."""
    if not os.path.exists(TRANSCRIPT_STORE_DIR):
        print("⚠️ Transcript store missing, rebuilding...")
        return build_transcript_vectorstore()

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        collection_name="transcript_store",
        persist_directory=TRANSCRIPT_STORE_DIR,
        embedding_function=embeddings
    )

    print("📄 Transcript retriever loaded.")
    return db


# =====================================================================
# 3. BUILD AWS KNOWLEDGE BASE VECTORSTORE
# =====================================================================

def build_aws_kb_vectorstore(kb_chunks):
    """
    kb_chunks = list of dicts:
        {
            "service": "...",
            "chunk_id": "...",
            "text": "..."
        }
    """

    timestamp = str(int(datetime.now().timestamp()))
    store_dir = os.path.join(AWS_STORE_BASE, f"store_{timestamp}")

    os.makedirs(store_dir, exist_ok=True)
    print(f"🛠 Creating AWS KB vectorstore at: {store_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        collection_name="aws_kb",
        persist_directory=store_dir,
        embedding_function=embeddings
    )

    docs = []
    for item in kb_chunks:
        docs.append(
            Document(
                page_content=item["text"],
                metadata={
                    "service": item["service"],
                    "chunk_id": item["chunk_id"]
                }
            )
        )

    db.add_documents(docs)

    # Meta file for debugging
    with open(os.path.join(store_dir, "meta.json"), "w", encoding="utf-8") as f:
        f.write(f'{{"timestamp": "{timestamp}"}}')

    print("✅ AWS KB vectorstore built.")
    return store_dir


# =====================================================================
# 4. LOAD LATEST AWS KB VECTORSTORE
# =====================================================================

def load_latest_aws_kb_vectorstore():
    """Load most recent AWS KB vectorstore version."""

    if not os.path.exists(AWS_STORE_BASE):
        print("⚠️ No AWS KB store found.")
        return None

    versions = sorted(glob.glob(os.path.join(AWS_STORE_BASE, "store_*")))
    if not versions:
        print("⚠️ No KB versions found.")
        return None

    latest = versions[-1]
    print(f"🔁 Loading AWS KB VectorStore from: {latest}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        collection_name="aws_kb",
        persist_directory=latest,
        embedding_function=embeddings
    )

    return db


# =====================================================================
# 5. RESET ALL STORES
# =====================================================================

def reset_all_vectorstores():
    print("🗑 Resetting all vectorstores...")

    if os.path.exists(TRANSCRIPT_STORE_DIR):
        shutil.rmtree(TRANSCRIPT_STORE_DIR)

    if os.path.exists(AWS_STORE_BASE):
        shutil.rmtree(AWS_STORE_BASE)

    print("✅ All vectorstores cleared.")


# =====================================================================
# DEBUG MODE
# =====================================================================

if __name__ == "__main__":
    print("Debug loading stores...")

    t = load_transcript_vectorstore()
    print("Transcript store:", t)

    a = load_latest_aws_kb_vectorstore()
    print("AWS KB store:", a)
