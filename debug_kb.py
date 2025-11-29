# --------------------------------------------------------------
# debug_kb.py  (Updated for new chunker + vectorstore system)
# --------------------------------------------------------------

from aws.aws_info import (
    rebuild_aws_kb_vectorstore,
    retrieve_from_kb,
)
from vector_store import load_latest_aws_kb_vectorstore

from kb_chunker import build_kb_chunks

# --------------------------------------------------------------

print("\n==============================")
print("🔍 DEBUG: AWS KB PIPELINE")
print("==============================\n")

# 1. Check chunks
kb_chunks = build_kb_chunks()

print(f"📄 Raw total chunks: {len(kb_chunks)}\n")
print("🔍 Showing first 3 chunks:\n")

for c in kb_chunks[:3]:
    print("Service:", c["service"])
    print("Chunk ID:", c["chunk_id"])
    print("Text Preview:", c["text"][:200], "...\n")

# 2. Rebuild vectorstore
print("\n🛠 Rebuilding KB vectorstore...")
rebuild_aws_kb_vectorstore()

# 3. Load latest store
db = load_latest_aws_kb_vectorstore()
if db is None:
    print("❌ ERROR: Could not load vectorstore.")
    exit()

print("\n📦 Vectorstore loaded successfully.\n")

# 4. Test a query
query = "what is s3"

print(f"🔎 Testing query: {query!r}\n")

doc, dist, err = retrieve_from_kb(query)

if err:
    print(err)
else:
    print(f"✅ BEST MATCH (distance={dist:.4f})")
    print(doc.page_content[:500])
