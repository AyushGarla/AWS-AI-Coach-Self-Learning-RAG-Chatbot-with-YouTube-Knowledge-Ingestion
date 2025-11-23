# --------------------------------------------------------------
# vector_store.py (FINAL, clean, stable)
# --------------------------------------------------------------
# Handles ONLY transcript vectorstores.
# Safe, overwrite-only, no versioning needed.
#
# Author: Ayush Garla
# --------------------------------------------------------------

import os
import shutil
import textwrap

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------
# Parse chunks from chunks.txt
# --------------------------------------------------------------
def load_chunks(chunks_path="chunks.txt"):
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"{chunks_path} missing. Run NLP pipeline first.")

    chunks, current = [], []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("### CHUNK "):
                if current:
                    chunks.append(" ".join(current))
                    current = []
                continue

            if line:
                current.append(line)

    if current:
        chunks.append(" ".join(current))

    return chunks


# --------------------------------------------------------------
# Build Transcript Vector Store (safe overwrite)
# --------------------------------------------------------------
def build_vector_store(chunks_path="chunks.txt", vector_dir="vectorstore"):
    """
    1. Removes OLD transcript vectorstore safely.
    2. Loads chunks from NLP pipeline.
    3. Builds NEW transcript vectorstore.
    """

    print(f"\n📦 Building new transcript vectorstore at '{vector_dir}'...")

    # -- remove old store safely --
    if os.path.exists(vector_dir):
        try:
            shutil.rmtree(vector_dir)
            print("🗑 Old transcript vectorstore removed.")
        except Exception as e:
            print(f"⚠️ Could not remove old vectorstore: {e}")

    # -- load chunks --
    chunks = load_chunks(chunks_path)
    print(f"✅ Loaded {len(chunks)} chunks.")

    print("\n🔎 Preview (first 200 chars):")
    print(textwrap.fill(chunks[0][:200], width=100))

    # -- embeddings --
    print("\n🔡 Loading embedding model:", EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # -- convert chunks to documents --
    docs = [
        Document(page_content=c, metadata={"chunk_id": i + 1})
        for i, c in enumerate(chunks)
    ]

    # -- build vectorstore --
    os.makedirs(vector_dir, exist_ok=True)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=vector_dir
    )

    print(f"\n📁 Transcript vectorstore created at: {vector_dir}")

    # Optional test
    example_q = "What is AWS Lambda?"
    results = db.similarity_search(example_q, k=1)
    print("\n🔍 Retrieval Test:")
    for r in results:
        print(f"- Chunk {r.metadata['chunk_id']}: {r.page_content[:150]}...")

    return db


# --------------------------------------------------------------
# Load existing transcript vectorstore
# --------------------------------------------------------------
def load_vector_store(vector_dir="vectorstore"):
    if not os.path.exists(vector_dir):
        raise FileNotFoundError(f"Transcript vectorstore '{vector_dir}' does not exist. Build it first.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"⬆️ Loading transcript vectorstore from: {vector_dir}")

    db = Chroma(
        persist_directory=vector_dir,
        embedding_function=embeddings
    )

    return db


# --------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    print("🏗 Building transcript vectorstore...\n")
    build_vector_store()
