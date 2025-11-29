# --------------------------------------------------------------
# aws_info.py  (FINAL — works with kb_chunker + vector_store)
# --------------------------------------------------------------
# Provides:
#   - Query normalization
#   - Checking if a topic exists in KB
#   - KB relevance scoring (cosine distance)
#   - Loading latest AWS KB vectorstore
#   - Rebuilding AWS KB vectorstore from markdown chunks
#   - Retrieve-from-KB logic for RAG_Agent
# --------------------------------------------------------------

import os
import json
import numpy as np

from kb_chunker import build_kb_chunks
from vector_store import (
    build_aws_kb_vectorstore,
    load_latest_aws_kb_vectorstore,
)

from langchain_huggingface import HuggingFaceEmbeddings

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Relevance threshold (cosine distance)
# Lower = more similar. 0.0 is perfect match.
RELEVANCE_THRESHOLD = 0.55   # good default


# --------------------------------------------------------------
# Normalize AWS topic
# --------------------------------------------------------------

def normalize_topic(topic: str) -> str:
    topic = topic.lower().strip()

    replacements = {
        "s3": "simple storage service",
        "ec2": "elastic compute cloud",
        "rds": "relational database service",
        "vpc": "virtual private cloud",
        "elb": "elastic load balancing",
        "alb": "application load balancer",
        "nlb": "network load balancer",
        "iam": "identity and access management",
        "sns": "simple notification service",
        "sqs": "simple queue service",
        "cf": "cloudformation",
        "cdk": "cloud development kit",
        "lambda": "lambda compute service",
    }

    for short, long in replacements.items():
        if topic == short or topic.startswith(short + " "):
            topic = long
            break

    if not topic.startswith("aws "):
        topic = "aws " + topic

    return topic


# --------------------------------------------------------------
# Basic cosine distance helper
# --------------------------------------------------------------

def cosine_distance(a, b):
    """Cosine distance; lower = more relevant."""
    a = np.array(a)
    b = np.array(b)

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 999

    return 1 - np.dot(a, b) / denom


# --------------------------------------------------------------
# Check KB relevance
# --------------------------------------------------------------

def kb_relevance_score(query: str, kb_vectorstore):
    """Returns (best_distance, best_doc, scored_results)."""

    query_emb = embeddings.embed_query(query)
    results = kb_vectorstore.similarity_search_with_score(query, k=5)

    scored = []
    for doc, _score in results:
        chunk_emb = embeddings.embed_query(doc.page_content)
        dist = cosine_distance(query_emb, chunk_emb)
        scored.append((dist, doc))

    scored.sort(key=lambda x: x[0])
    best_dist, best_doc = scored[0]

    return best_dist, best_doc, scored


# --------------------------------------------------------------
# Public function: Retrieve a topic from AWS KB
# --------------------------------------------------------------

def retrieve_from_kb(query: str):
    kb = load_latest_aws_kb_vectorstore()

    if kb is None:
        return None, None, "❌ No AWS KB vectorstore found."

    normalized = normalize_topic(query)

    dist, best_doc, scored = kb_relevance_score(normalized, kb)

    print(f"\n🔎 Cosine distance for '{normalized}': {dist:.4f}")
    print(f"→ Preview: {best_doc.page_content[:150]}...\n")

    if dist <= RELEVANCE_THRESHOLD:
        return best_doc, dist, None

    return None, dist, f"❌ Topic '{normalized}' not found in KB."


# --------------------------------------------------------------
# Check whether topic exists in KB (before fetching)
# --------------------------------------------------------------

def is_topic_in_aws_kb(topic: str) -> bool:
    kb = load_latest_aws_kb_vectorstore()
    if kb is None:
        return False

    normalized = normalize_topic(topic)
    dist, doc, _ = kb_relevance_score(normalized, kb)

    return dist <= RELEVANCE_THRESHOLD


# --------------------------------------------------------------
# Append new learned text to KB file
# --------------------------------------------------------------

KB_FILE_PATH = "aws/data/aws_knowledge_base.txt"

def append_to_aws_kb(topic: str, text: str):
    """Adds a new section to KB text file with proper markdown."""
    with open(KB_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n## " + topic + "\n")
        f.write(text.strip() + "\n")

    print(f"📌 Appended new topic to KB: {topic}")


# --------------------------------------------------------------
# Rebuild KB vectorstore
# --------------------------------------------------------------

def rebuild_aws_kb_vectorstore():
    print("🛠 Rebuilding AWS KB vectorstore using kb_chunker...")

    kb_chunks = build_kb_chunks()
    print(f"🧩 Total KB chunks: {len(kb_chunks)}")

    store_dir = build_aws_kb_vectorstore(kb_chunks)

    print(f"🎉 AWS KB vectorstore rebuilt at: {store_dir}")
    return store_dir


# --------------------------------------------------------------
# Teaching helper
# --------------------------------------------------------------

def learn_new_topic(topic: str, text: str):
    """Add → Rebuild → Confirm."""
    append_to_aws_kb(topic, text)
    rebuild_aws_kb_vectorstore()
    return f"✅ Learned new topic: {topic}"


# --------------------------------------------------------------
# Debug entry point
# --------------------------------------------------------------

if __name__ == "__main__":
    print("Debug: rebuilding KB...")
    rebuild_aws_kb_vectorstore()
