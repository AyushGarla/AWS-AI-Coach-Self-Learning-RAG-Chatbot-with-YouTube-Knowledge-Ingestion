# --------------------------------------------------------------
# RAG_Agent.py  (FINAL — Correct quiz generation + no placeholders)
# --------------------------------------------------------------

import os
import json
import re
import textwrap
from typing import List, Dict, Any

from dotenv import load_dotenv

from aws.aws_info import (
    get_aws_retriever,
    is_topic_in_aws_kb,
    load_latest_vectorstore
)

from vector_store import load_vector_store

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --------------------------------------------------------------
# API Key
# --------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY")
print("🔐 OpenAI API Key loaded.")


# --------------------------------------------------------------
# Transcript retriever
# --------------------------------------------------------------
def load_transcript_retriever():
    try:
        T_DB = load_vector_store("vectorstore")
        print("📄 Transcript retriever loaded.")
        return T_DB.as_retriever(search_kwargs={"k": 3})
    except Exception:
        print("⚠️ No transcript vectorstore found.")
        return None


TRANSCRIPT_RETRIEVER = load_transcript_retriever()


# --------------------------------------------------------------
# Helper: Format retrieved docs
# --------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(
        textwrap.shorten(d.page_content, width=600, placeholder="…")
        for d in docs
    )


# --------------------------------------------------------------
# Prompts
# --------------------------------------------------------------
TEACH_TMPL = """
You are an AWS Coach.

Use the context to answer clearly and step-by-step
for a BEGINNER.

If the context does NOT contain the answer, reply:
"I don't see that in my available knowledge."

Context:
{context}

Question:
{question}

Explain in simple, friendly language.
"""
teach_prompt = ChatPromptTemplate.from_template(TEACH_TMPL)
teach_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
teach_parser = StrOutputParser()


# ✅ IMPORTANT: use REAL variables {topic}, {num}, {context}
# Escape ONLY JSON braces in the example.
QUIZ_TMPL = """
You are an AWS Quiz Master.

Create {num} beginner-friendly multiple-choice questions (MCQs)
about the topic: "{topic}"

Rules:
- Questions MUST be about the given AWS topic.
- Prefer using the context if possible.
- Exactly 4 options: A, B, C, D.
- Only ONE correct answer.
- Include a short explanation for the correct answer.
- Output MUST be a strict JSON list only (no extra text).

Example format:
[
  {{
    "question": "....?",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "answer": "A",
    "explanation": "..."
  }}
]

Context:
{context}
"""
quiz_prompt = ChatPromptTemplate.from_template(QUIZ_TMPL)
quiz_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.15)
quiz_parser = StrOutputParser()


CODE_TMPL = """
You are an AWS Code Helper.

Provide:
1) A short beginner explanation.
2) A {lang} code snippet.
3) A short explanation of the code.

Use ONLY this context:

{context}

Question:
{question}
"""
code_prompt = ChatPromptTemplate.from_template(CODE_TMPL)
code_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25)
code_parser = StrOutputParser()


# --------------------------------------------------------------
# Intent keywords
# --------------------------------------------------------------
QUIZ_KEYWORDS = [
    "quiz", "test", "mcq", "practice",
    "questions on", "question on", "question",
    "ask questions", "ask question",
    "ask me questions", "give questions",
    "prepare questions", "question me"
]

CODE_KEYWORDS = ["code", "python", "sql", "script", "boto3"]


# --------------------------------------------------------------
# Normalize query (strip intent phrases + add AWS prefix)
# --------------------------------------------------------------
def normalize_query(q: str) -> str:
    original = q.strip()
    lowered = original.lower()

    # remove quiz-leading phrases
    for kw in sorted(QUIZ_KEYWORDS, key=len, reverse=True):
        if lowered.startswith(kw):
            original = original[len(kw):].strip()
            lowered = original.lower()
            break

    # remove leftover "questions"/"question"
    for prefix in ["questions", "question"]:
        if lowered.startswith(prefix):
            original = original[len(prefix):].strip()
            lowered = original.lower()

    # add AWS prefix if missing
    if original and not lowered.startswith("aws"):
        original = "AWS " + original

    print(f"🛠 Normalized Query → {original}")
    return original


# --------------------------------------------------------------
# Intent Classification
# --------------------------------------------------------------
def classify_intent(q: str) -> str:
    text = q.lower()

    if any(kw in text for kw in QUIZ_KEYWORDS):
        return "QUIZ"
    if any(kw in text for kw in CODE_KEYWORDS):
        return "CODE"
    return "TEACH"


# --------------------------------------------------------------
# Retriever selection (AWS-first)
# --------------------------------------------------------------
def choose_retriever(query: str):
    load_latest_vectorstore()  # always refresh AWS store

    if is_topic_in_aws_kb(query):
        print("📚 Using AWS KB\n")
        return get_aws_retriever(), "aws"

    print("⚠️ AWS KB insufficient. Using transcript if available.\n")
    return TRANSCRIPT_RETRIEVER, "transcript"


# --------------------------------------------------------------
# TEACH MODE
# --------------------------------------------------------------
def teach(query: str) -> str:
    retriever, source = choose_retriever(query)

    if retriever is None:
        msg = "❌ I don't have information yet. Provide a YouTube URL."
        print(msg)
        return msg

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | teach_prompt
        | teach_llm
        | teach_parser
    )

    print(f"🎭 TEACH MODE (Source: {source.upper()})")
    answer = chain.invoke(query)

    print("\n📘 Explanation:\n")
    print(answer)

    return answer


# --------------------------------------------------------------
# QUIZ MODE (robust JSON parsing)
# --------------------------------------------------------------
def extract_json_list(s: str):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\[\s*\{.*?\}\s*\]", s, flags=re.S)
        if not m:
            raise ValueError("Could not parse MCQ JSON output.")
        return json.loads(m.group(0))


def generate_quiz(topic: str, num: int = 3):
    retriever, source = choose_retriever(topic)

    if retriever is None:
        print("❌ No info for quiz. Need YouTube first.")
        return []

    docs = retriever.invoke(topic)
    context = format_docs(docs)

    chain = (
        {
            "context": lambda _: context,
            "topic": lambda _: topic,
            "num": lambda _: num
        }
        | quiz_prompt
        | quiz_llm
        | quiz_parser
    )

    raw = chain.invoke({})
    data = extract_json_list(raw)

    quiz = []
    if not isinstance(data, list):
        return quiz

    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            quiz.append({
                "question": str(item["question"]).strip(),
                "options": {
                    "A": str(item["options"]["A"]).strip(),
                    "B": str(item["options"]["B"]).strip(),
                    "C": str(item["options"]["C"]).strip(),
                    "D": str(item["options"]["D"]).strip(),
                },
                "answer": str(item["answer"]).strip().upper(),
                "explanation": str(item.get("explanation", "")).strip(),
            })
        except Exception:
            continue

    return quiz


def print_quiz(quiz):
    print("\n📝 AWS Beginner Quiz:\n")
    for i, q in enumerate(quiz, 1):
        print(f"Q{i}. {q['question']}")
        for k, v in q["options"].items():
            print(f"   {k}) {v}")


def grade_quiz(quiz, responses):
    score = 0
    report = []

    for i, (q, r) in enumerate(zip(quiz, responses), 1):
        correct = q["answer"]
        user_ans = r.upper()
        ok = (user_ans == correct)
        score += int(ok)

        report.append({
            "q": i,
            "you": user_ans,
            "correct": correct,
            "ok": ok,
            "explanation": q["explanation"]
        })

    return score, report


# --------------------------------------------------------------
# CODE HELPER MODE
# --------------------------------------------------------------
def generate_code_answer(q: str, lang="Python") -> str:
    retriever, source = choose_retriever(q)
    if retriever is None:
        return "❌ Need YouTube link first."

    docs = retriever.invoke(q)
    ctx = format_docs(docs)

    chain = (
        {
            "context": lambda _: ctx,
            "question": lambda _: q,
            "lang": lambda _: lang,
        }
        | code_prompt
        | code_llm
        | code_parser
    )

    out = chain.invoke({})
    print("\n💻 Code Helper:\n")
    print(out)
    return out


# --------------------------------------------------------------
# Main Router
# --------------------------------------------------------------
def ai_coach(q: str):
    intent = classify_intent(q)
    normalized = normalize_query(q)

    if intent == "QUIZ":
        quiz = generate_quiz(normalized, num=3)
        if not quiz:
            return "❌ No quiz available."

        print_quiz(quiz)
        print("\n💡 Enter answers A/B/C/D:")

        answers = []
        for i in range(len(quiz)):
            ans = input(f"Answer Q{i+1}: ").strip().upper()
            while ans not in ["A", "B", "C", "D"]:
                ans = input("Enter A/B/C/D: ").strip().upper()
            answers.append(ans)

        score, rep = grade_quiz(quiz, answers)
        print(f"\n🏁 Score: {score}/{len(quiz)}\n")

        for r in rep:
            icon = "✅" if r["ok"] else "❌"
            print(f"{icon} Q{r['q']} — You: {r['you']} | Correct: {r['correct']}")
            print(f"   💬 {r['explanation']}\n")
        return

    if intent == "CODE":
        return generate_code_answer(normalized)

    return teach(normalized)


# --------------------------------------------------------------
# Debug
# --------------------------------------------------------------
if __name__ == "__main__":
    ai_coach("questions on S3")
