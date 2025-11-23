# --------------------------------------------------------------
# app.py  (Streamlit UI for AWS AI Coach)
# --------------------------------------------------------------
# Friendly chatbot UI for:
# 1) Teach / Q&A
# 2) Quiz mode (3 MCQs + validate + feedback)
# 3) Code helper
# 4) Learn from YouTube (self-learning loop)
#
# Uses your existing pipeline:
# - transcript_extraction.py
# - NLP.py
# - vector_store.py
# - aws/aws_info.py
# - RAG_Agent.py
#
# Author: Ayush Garla + ChatGPT UI layer
# --------------------------------------------------------------

import streamlit as st
import importlib
from datetime import datetime

# Your pipeline imports
import RAG_Agent
from transcript_extraction import get_youtube_transcript
from NLP import run_nlp_pipeline
from vector_store import build_vector_store
from aws.aws_info import (
    is_topic_in_aws_kb,
    append_to_aws_kb,
    rebuild_aws_kb_vectorstore,
    load_latest_vectorstore,
)

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------
st.set_page_config(
    page_title="AWS AI Coach",
    page_icon="☁️",
    layout="wide"
)

# --------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of dicts {role, content, ts}
    if "mode" not in st.session_state:
        st.session_state.mode = "TEACH"  # TEACH | QUIZ | CODE | YOUTUBE
    if "pending_quiz" not in st.session_state:
        st.session_state.pending_quiz = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = []
    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = ""
    if "last_teach_answer" not in st.session_state:
        st.session_state.last_teach_answer = ""
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False

def add_chat(role, content):
    st.session_state.chat.append({
        "role": role,
        "content": content,
        "ts": datetime.now().strftime("%H:%M")
    })

def render_chat():
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def refresh_rag_agent():
    # make sure retrievers reload after new vectorstores
    importlib.reload(RAG_Agent)

# --------------------------------------------------------------
# Initialize session
# --------------------------------------------------------------
init_state()

# --------------------------------------------------------------
# Sidebar / navigation
# --------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☁️ AWS AI Coach")
    st.caption("Train for AWS interviews + exams with a friendly AI coach.")

    st.divider()

    st.markdown("### 🎯 Mode")
    mode = st.radio(
        label="Choose how you want to train",
        options=[
            "📘 Teach / Q&A",
            "📝 Quiz (MCQs)",
            "💻 Code Helper",
            "🎥 Learn from YouTube"
        ]
    )

    if mode.startswith("📘"):
        st.session_state.mode = "TEACH"
    elif mode.startswith("📝"):
        st.session_state.mode = "QUIZ"
    elif mode.startswith("💻"):
        st.session_state.mode = "CODE"
    else:
        st.session_state.mode = "YOUTUBE"

    st.divider()

    st.markdown("### ⚙️ KB Tools")
    if st.button("🔄 Reload latest AWS KB"):
        with st.spinner("Loading latest AWS KB vectorstore..."):
            load_latest_vectorstore()
            refresh_rag_agent()
            st.session_state.kb_loaded = True
        st.success("Latest AWS KB loaded!")

    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.session_state.pending_quiz = None
        st.session_state.quiz_answers = []
        st.session_state.quiz_topic = ""
        st.rerun()


# --------------------------------------------------------------
# Header
# --------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;">
        <h1 style="margin:0;">AWS AI Coach</h1>
        <span style="font-size:24px;">☁️</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("A friendly chatbot to help you master AWS: explain topics, answer questions, quiz you, and learn from YouTube.")

# Load KB once per session automatically
if not st.session_state.kb_loaded:
    with st.spinner("Loading latest AWS KB..."):
        load_latest_vectorstore()
        refresh_rag_agent()
    st.session_state.kb_loaded = True

# --------------------------------------------------------------
# Chat area
# --------------------------------------------------------------
render_chat()

# --------------------------------------------------------------
# Mode: Teach / Q&A
# --------------------------------------------------------------
if st.session_state.mode == "TEACH":
    st.info("📘 Teach / Q&A mode: ask any AWS topic or question. Example: *AWS Glue*, *What is S3 Lifecycle?*")

    user_q = st.chat_input("Ask an AWS topic or question...")

    if user_q:
        add_chat("user", user_q)

        norm_q = RAG_Agent.normalize_query(user_q)
        intent = RAG_Agent.classify_intent(user_q)

        # If user mistakenly asks quiz/code here, route automatically
        if intent == "QUIZ":
            st.session_state.mode = "QUIZ"
            st.rerun()
        if intent == "CODE":
            st.session_state.mode = "CODE"
            st.rerun()

        with st.spinner("Thinking..."):
            if not is_topic_in_aws_kb(norm_q):
                add_chat(
                    "assistant",
                    f"⚠️ I don’t have this topic yet in my AWS KB.\n\n"
                    f"👉 Switch to **Learn from YouTube** mode to teach me this topic."
                )
            else:
                answer = RAG_Agent.teach(norm_q)
                st.session_state.last_teach_answer = answer
                add_chat("assistant", answer)

        st.rerun()


# --------------------------------------------------------------
# Mode: Quiz
# --------------------------------------------------------------
elif st.session_state.mode == "QUIZ":
    st.warning("📝 Quiz mode: type a topic like *AWS S3* or *Glue* and I’ll generate 3 MCQs.")

    # Step 1: if no quiz pending, ask for topic
    if st.session_state.pending_quiz is None:
        quiz_topic_input = st.chat_input("Enter a topic for quiz (e.g., AWS Glue)...")

        if quiz_topic_input:
            add_chat("user", f"Quiz on: **{quiz_topic_input}**")

            topic = RAG_Agent.normalize_query(quiz_topic_input)

            with st.spinner("Preparing your quiz..."):
                if not is_topic_in_aws_kb(topic):
                    add_chat(
                        "assistant",
                        f"⚠️ I don’t have **{topic}** in KB yet.\n\n"
                        f"👉 Go to **Learn from YouTube** mode to add it first."
                    )
                else:
                    quiz = RAG_Agent.generate_quiz(topic, num=3)
                    if not quiz:
                        add_chat("assistant", "❌ I couldn’t generate a quiz. Try another topic.")
                    else:
                        st.session_state.pending_quiz = quiz
                        st.session_state.quiz_answers = ["", "", ""]
                        st.session_state.quiz_topic = topic
                        add_chat("assistant", f"✅ Quiz ready for **{topic}**! Scroll down to answer.")

            st.rerun()

    # Step 2: show quiz questions if pending
    else:
        quiz = st.session_state.pending_quiz
        st.markdown(f"### 🧠 Quiz on **{st.session_state.quiz_topic}**")

        # Show questions with radio options
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            st.session_state.quiz_answers[i] = st.radio(
                label=f"Select answer for Q{i+1}",
                options=["A", "B", "C", "D"],
                format_func=lambda x: f"{x}) {q['options'][x]}",
                key=f"quiz_{i}"
            )
            st.write("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Submit Quiz"):
                answers = st.session_state.quiz_answers
                score, report = RAG_Agent.grade_quiz(quiz, answers)

                feedback_md = f"🏁 **Score: {score}/{len(quiz)}**\n\n"
                for r in report:
                    icon = "✅" if r["ok"] else "❌"
                    feedback_md += (
                        f"{icon} **Q{r['q']}** — You: **{r['you']}** | Correct: **{r['correct']}**\n\n"
                        f"> {r['explanation']}\n\n"
                    )

                add_chat("assistant", feedback_md)

                # Reset quiz state
                st.session_state.pending_quiz = None
                st.session_state.quiz_answers = []
                st.session_state.quiz_topic = ""

                st.rerun()

        with col2:
            if st.button("↩️ Cancel Quiz"):
                st.session_state.pending_quiz = None
                st.session_state.quiz_answers = []
                st.session_state.quiz_topic = ""
                add_chat("assistant", "↩️ Quiz cancelled. Pick a new topic anytime.")
                st.rerun()


# --------------------------------------------------------------
# Mode: Code Helper
# --------------------------------------------------------------
elif st.session_state.mode == "CODE":
    st.success("💻 Code Helper mode: ask for AWS code help (Python/boto3/SQL).")

    code_q = st.chat_input("Ask for AWS code help… e.g., 'code for S3 upload in boto3'")

    if code_q:
        add_chat("user", code_q)

        norm_q = RAG_Agent.normalize_query(code_q)

        with st.spinner("Generating code…"):
            if not is_topic_in_aws_kb(norm_q):
                add_chat(
                    "assistant",
                    f"⚠️ I don’t have KB info on **{norm_q}** yet.\n\n"
                    f"👉 Learn it first via **Learn from YouTube** mode."
                )
            else:
                out = RAG_Agent.generate_code_answer(norm_q)
                add_chat("assistant", out)

        st.rerun()


# --------------------------------------------------------------
# Mode: Learn from YouTube
# --------------------------------------------------------------
else:
    st.markdown("### 🎥 Learn from YouTube")
    st.caption("Paste a YouTube link, give the topic, and I’ll learn it into my AWS KB.")

    with st.form("yt_form"):
        topic_in = st.text_input("AWS topic to learn (example: AWS Step Functions)")
        url_in = st.text_input("YouTube URL")
        submitted = st.form_submit_button("🚀 Learn this Topic")

    if submitted:
        if not topic_in or not url_in:
            st.error("Please provide both topic and YouTube URL.")
        else:
            add_chat("user", f"Learn from YouTube → **{topic_in}**\n\n{url_in}")

            topic = RAG_Agent.normalize_query(topic_in)

            with st.spinner("1/6 Extracting transcript…"):
                get_youtube_transcript(url_in)

            with st.spinner("2/6 NLP preprocessing…"):
                run_nlp_pipeline()

            with st.spinner("3/6 Building transcript vectorstore…"):
                build_vector_store()

            refresh_rag_agent()

            with st.spinner("4/6 Generating explanation from transcript…"):
                explanation = RAG_Agent.teach(topic)

            with st.spinner("5/6 Appending to KB…"):
                append_to_aws_kb(topic, explanation)

            with st.spinner("6/6 Building NEW KB vectorstore version…"):
                rebuild_aws_kb_vectorstore()
                load_latest_vectorstore()
                refresh_rag_agent()

            add_chat(
                "assistant",
                f"✅ Learned **{topic}** and added it to AWS Knowledge Base!\n\n"
                f"Now you can ask me about this topic in Teach / Quiz / Code modes."
            )

            st.success("Done! KB updated.")
            st.rerun()
