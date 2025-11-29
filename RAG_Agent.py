# --------------------------------------------------------------
# RAG_Agent.py (FINAL — Fully Compatible with New Pipeline)
# --------------------------------------------------------------

import os
import json
import re
import textwrap
from typing import Dict, Any

from dotenv import load_dotenv

# AWS KB relevance + retriever
from aws.aws_info import (
    retrieve_from_kb,
    normalize_topic
)

# Transcript retrieval
from vector_store import load_transcript_vectorstore

# LLMs
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------------------
# ENV + MODEL INIT
# --------------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY missing in dotenv or environment!")

print("🔐 OpenAI API Key loaded.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# --------------------------------------------------------------
# TRANSCRIPT RETRIEVER
# --------------------------------------------------------------
def load_transcript_retriever():
    try:
        db = load_transcript_vectorstore()
        return db.as_retriever(search_kwargs={"k": 3})
    except:
        return None


TRANSCRIPT_RETRIEVER = load_transcript_retriever()


# --------------------------------------------------------------
# Format docs
# --------------------------------------------------------------
def format_docs(docs, max_len=600):
    if not docs:
        return ""
    return "\n\n".join(
        textwrap.shorten(d.page_content, width=max_len, placeholder="…")
        for d in docs
    )


# --------------------------------------------------------------
# Intent Classification
# --------------------------------------------------------------
QUIZ_KEYWORDS = ["quiz", "test", "mcq", "practice", "questions"]
CODE_KEYWORDS = ["code", "script", "python", "boto3"]


def classify_intent(q: str):
    ql = q.lower()
    if any(k in ql for k in QUIZ_KEYWORDS):
        return "QUIZ"
    if any(k in ql for k in CODE_KEYWORDS):
        return "CODE"
    return "TEACH"


# --------------------------------------------------------------
# Query Normalization
# --------------------------------------------------------------
def normalize_query(q: str):
    q = q.strip().lower()
    if not q.startswith("aws "):
        q = "aws " + q
    print(f"🛠 Normalized → {q}")
    return q


# --------------------------------------------------------------
# HYBRID RETRIEVAL (AWS KB + Transcript)
# --------------------------------------------------------------
def hybrid_retrieve(query: str, mode="teach"):
    """
    TEACH → AWS KB relevance + transcript fallback
    QUIZ/CODE → ignore relevance filtering
    """

    # AWS KB (for teach mode only)
    kb_doc = None
    if mode == "teach":
        kb_doc, dist, err = retrieve_from_kb(query)
        if kb_doc:
            print("📚 Using AWS KB for TEACH")
            return [("aws", kb_doc)]

    # Transcript
    tr_docs = []
    if TRANSCRIPT_RETRIEVER:
        try:
            tr_docs = TRANSCRIPT_RETRIEVER.get_relevant_documents(query)
        except:
            pass

    merged = []
    for d in tr_docs:
        merged.append(("transcript", d))

    return merged[:3]


# --------------------------------------------------------------
# TEACH PROMPT
# --------------------------------------------------------------
TEACH_TMPL = """
You are an AWS Coach.

Use the context below **only if relevant**.
If context does not contain the answer, say:
"I don't see that fully in my current knowledge."

Context:
{context}

Question:
{question}
"""

teach_prompt = ChatPromptTemplate.from_template(TEACH_TMPL)
teach_parser = StrOutputParser()


def teach(query: str):
    results = hybrid_retrieve(query, mode="teach")
    ctx = format_docs([d for _, d in results])

    chain = (
        {"context": lambda _: ctx, "question": RunnablePassthrough()}
        | teach_prompt
        | llm
        | teach_parser
    )

    print("🎭 TEACH MODE → hybrid retrieval used")
    return chain.invoke(query)


# --------------------------------------------------------------
# QUIZ GENERATION
# --------------------------------------------------------------
QUIZ_TMPL = """
You are an AWS Quiz Master.
Create exactly 3 MCQs for beginners on: "{topic}"

RULES:
- 4 options: A, B, C, D
- EXACTLY ONE correct answer labeled under "answer"
- Provide a short explanation
- Output strictly a JSON list

Context:
{context}
"""

quiz_prompt = ChatPromptTemplate.from_template(QUIZ_TMPL)
quiz_parser = StrOutputParser()


def extract_json_list_quiz(output: str):
    try:
        data = json.loads(output)
        if isinstance(data, list):
            return data
    except:
        pass

    m = re.search(r"\[\s*\{.*?\}\s*\]", output, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            return []

    return []


def generate_quiz(topic: str, num=3):
    results = hybrid_retrieve(topic, mode="quiz")
    ctx = format_docs([d for _, d in results])

    chain = (
        {"context": lambda _: ctx, "topic": lambda _: topic}
        | quiz_prompt
        | llm
        | quiz_parser
    )

    raw = chain.invoke({})

    data = extract_json_list_quiz(raw)
    if not data:
        raise ValueError("❌ Could not parse MCQ JSON output.")

    quiz = []
    for item in data:
        q_text = item.get("question", "").strip()
        opts = item.get("options", {})
        ans = item.get("answer", "").strip().upper()
        expl = item.get("explanation", "").strip()

        if ans not in ["A", "B", "C", "D"]:
            ans = "A"
        if not expl:
            expl = "No explanation provided."

        clean_opts = {k.upper(): v.strip() for k, v in opts.items()}

        quiz.append({
            "question": q_text,
            "options": clean_opts,
            "answer": ans,
            "explanation": expl,
        })

    return quiz


# --------------------------------------------------------------
# QUIZ GRADING
# --------------------------------------------------------------
def grade_quiz(quiz, responses):
    score = 0
    report = []

    for i, (q, r) in enumerate(zip(quiz, responses), 1):
        ok = (r.upper() == q["answer"])
        score += int(ok)
        report.append({
            "q": i,
            "you": r.upper(),
            "correct": q["answer"],
            "ok": ok,
            "explanation": q["explanation"]
        })

    return score, report


# --------------------------------------------------------------
# CODE MODE
# --------------------------------------------------------------
CODE_TMPL = """
You are an AWS Code Helper.

Provide:
1) Short explanation
2) A {lang} code snippet
3) Explanation of code

Context:
{context}

Question:
{question}
"""

code_prompt = ChatPromptTemplate.from_template(CODE_TMPL)
code_parser = StrOutputParser()


def generate_code_answer(q: str, lang="Python"):
    results = hybrid_retrieve(q, mode="code")
    ctx = format_docs([d for _, d in results])

    chain = (
        {
            "context": lambda _: ctx,
            "question": lambda _: q,
            "lang": lambda _: lang,
        }
        | code_prompt
        | llm
        | code_parser
    )

    return chain.invoke({})


# --------------------------------------------------------------
# DEBUG
# --------------------------------------------------------------
if __name__ == "__main__":
    print(teach("what is s3"))
