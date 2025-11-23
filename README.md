📘 AWS AI Coach — Self-Learning RAG Chatbot

An intelligent AWS training assistant powered by Retrieval-Augmented Generation (RAG) with self-learning capabilities from YouTube videos.
It can teach AWS concepts, quiz you, generate code, and update its own knowledge base automatically.

Built using Streamlit, LangChain, OpenAI, ChromaDB, Whisper ASR, yt-dlp, and SpaCy.

🚀 Features
✅ Teach Mode

Clear, beginner-friendly explanations using AWS Knowledge Base + Semantic Retrieval.

✅ Quiz Mode

Auto-generates 3 MCQs for any AWS topic → evaluates your answers → gives detailed feedback.

✅ Code Helper Mode

Generates AWS Python (boto3) code snippets with explanations.

✅ Self-Learning Mode

If a topic is unknown:

Asks for a YouTube link

Extracts subtitles or uses Whisper AI

Cleans + chunks transcript

Builds transcript vectorstore

Generates AWS-style explanation

Adds it permanently to AWS KB

Creates a new vectorstore version

✅ Streamlit Chatbot UI

Modern, student-friendly, interactive interface.

📁 Project Structure
Final_Project_RAG/
│
├── main.py                     # CLI orchestrator (learning + routing)
├── RAG_Agent.py                # Teach / Quiz / Code logic
├── transcript_extraction.py    # YouTube subtitles + Whisper ASR
├── NLP.py                      # Cleaning + chunking
├── vector_store.py             # Transcript vectorstore
│
├── aws/
│   ├── data/
│   │   └── aws_knowledge_base.txt
│   └── vectorstore_versions/   # Versioned KB stores
│
├── vectorstore/                # Transcript store
│
├── streamlit_app.py            # Interactive chatbot UI
├── .env                        # API keys
└── requirements.txt

🔑 Environment Setup

Create your .env file:

OPENAI_API_KEY=your_openai_key_here

🛠️ Install Requirements
pip install -r requirements.txt


Install SpaCy model:

python -m spacy download en_core_web_sm

🧠 How the System Works
1️⃣ User Query → Intent Detection

The system identifies whether the user wants:

Teach

Quiz

Code

Self-Learning

2️⃣ Normalize Query

Removes unwanted prefixes like:

quiz on, ask questions on, question, questions, practice


Ensures AWS prefix is added:

"glue" → "AWS Glue"

3️⃣ AWS KB Retrieval

System loads the latest versioned vectorstore:

aws/vectorstore_versions/store_<timestamp>


Checks relevance using semantic similarity.

If relevant → Use KB retriever.
If not → Activate Self-Learning Mode.

📚 Teach Mode (RAG Explanation)

Pipeline:

Query normalized

AWS KB → retrieve top-k chunks

Formatted context passed to OpenAI

TEACH prompt generates a clear explanation

Returned to user

📝 Quiz Mode (MCQs)

Detect quiz intent

Get context from retriever

LLM generates strict JSON MCQs

Display 3 questions

User inputs answers

System validates + shows:

Correct / Wrong

Correct Answer

Explanation

💻 Code Helper Mode

Generates AWS code:

boto3

IAM setup

Lambda functions

S3 uploads

DynamoDB operations

Returns:

Beginner explanation

Code snippet

Breakdown of functionality

🤖 Self-Learning Mode (YouTube → Knowledge Base)

If a topic is missing:

Step 1 — Ask for YouTube link

User enters URL.

Step 2 — Extract transcript

Try YouTube subtitles

Else use Whisper AI

Step 3 — NLP Processing

Clean transcript

Sentence splitting (SpaCy)

Chunk into ~500 word blocks

Step 4 — Build transcript vectorstore

Stored in:

/vectorstore

Step 5 — RAG explanation

Teach mode explanation is generated based on transcript chunks.

Step 6 — Persist Knowledge

Explanation appended to:

aws/data/aws_knowledge_base.txt

Step 7 — New Vectorstore Version

Automatically creates:

aws/vectorstore_versions/store_<new_timestamp>/

🎨 Running the Streamlit UI
streamlit run streamlit_app.py


This opens an intuitive interface where users can:

Ask AWS questions

Generate quizzes

Get code examples

Teach the assistant NEW topics through YouTube

▶️ Running the CLI Version

If you prefer terminal mode:

python main.py

🧪 Testing the Model

Try:

Explain AWS Glue
quiz on S3
give questions on Lambda
python code for S3 upload
learn VPC from YouTube

📦 Requirements

(These match your final working system)

streamlit
langchain
langchain-core
langchain-community
langchain-openai
chromadb
sentence-transformers
spacy
spacy-transformers
yt-dlp
openai
python-dotenv
torch
transformers
tqdm
numpy
pandas
regex


SpaCy model:

python -m spacy download en_core_web_sm

🏁 Conclusion

This project delivers:

A complete self-learning AI tutor for AWS

Versioned knowledge-base system

RAG-powered explanations

Automatic YouTube → transcript → vectorstore ingestion

MCQ generation + scoring

Code helper

Fully interactive Streamlit UI

Perfect for:

Students

AWS beginners

Trainers

AI/ML portfolio showcase
