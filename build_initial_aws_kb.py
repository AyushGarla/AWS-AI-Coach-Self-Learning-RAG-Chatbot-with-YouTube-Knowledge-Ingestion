# build_initial_aws_kb.py

from aws.aws_info import rebuild_aws_kb_vectorstore

print("🏗 Building AWS Knowledge Base vector store...")
rebuild_aws_kb_vectorstore()
print("✅ Done! Now you can run main.py")
