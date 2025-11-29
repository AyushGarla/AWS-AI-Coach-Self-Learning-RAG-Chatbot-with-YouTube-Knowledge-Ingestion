# --------------------------------------------------------------
# app.py  (Streamlit UI for AWS AI Coach — New KB Pipeline)
# --------------------------------------------------------------
# Modes:
# 1) Auto: intent-based (Teach / Quiz / Code)
# 2) Manual: Teach / Quiz / Code / Learn from YouTube
#
# Behavior:
# - TEACH:
#       • Check AWS KB via retrieve_from_kb()
#       • If missing → ask for YouTube URL → learn → append to KB → rebuild KB store
# - QUIZ:
#       • Pure LLM-based quiz (NO KB relevance checks)
# - CODE:
#       • Direct LLM-based code helper (optionally using hybrid retrieval inside RAG_Agent)
# --------------------------------------------------------------

import streamlit as st
import importlib
from datetime import datetime

import RAG_Agent
from transcript_extraction import get_youtube_transcript
from NLP import run_nlp_pipeline

from vector_store import (
    build_transcript_vectorstore,
    load_latest_aws_kb_vectorstore,
)

from aws.aws_info import (
    retrieve_from_kb,
    append_to_aws_kb,
    rebuild_aws_kb_vectorstore,
    normalize_topic,
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
# Session State
# --------------------------------------------------------------
def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of {role, content, ts}
    if "mode" not in st.session_state:
        st.session_state.mode = "TEACH"  # TEACH, QUIZ, CODE, YOUTUBE
    if "mode_setting" not in st.session_state:
        st.session_state.mode_setting = "AUTO"  # AUTO or MANUAL
    if "pending_quiz" not in st.session_state:
        st.session_state.pending_quiz = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = []
    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = ""
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False
    if "learning_topic" not in st.session_state:
        st.session_state.learning_topic = None  # topic we need YouTube URL for


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
    importlib.reload(RAG_Agent)


# --------------------------------------------------------------
# QUIZ UI (uses LLM-only quiz; no KB relevance checks)
# --------------------------------------------------------------
def render_quiz_ui():
    quiz = st.session_state.pending_quiz
    if not quiz:
        return

    st.markdown(f"### 🧠 Quiz on **{st.session_state.quiz_topic}**")

    if len(st.session_state.quiz_answers) != len(quiz):
        st.session_state.quiz_answers = [""] * len(quiz)

    for i, q in enumerate(quiz):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        st.session_state.quiz_answers[i] = st.radio(
            label=f"Select answer for Q{i+1}",
            options=["A", "B", "C", "D"],
            format_func=lambda x, opts=q["options"]: f"{x}) {opts[x]}",
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

            st.session_state.pending_quiz = None
            st.session_state.quiz_answers = []
            st.session_state.quiz_topic = ""
            st.session_state.mode = "TEACH"
            st.rerun()

    with col2:
        if st.button("↩️ Cancel Quiz"):
            st.session_state.pending_quiz = None
            st.session_state.quiz_answers = []
            st.session_state.quiz_topic = ""
            add_chat("assistant", "↩️ Quiz cancelled.")
            st.session_state.mode = "TEACH"
            st.rerun()


# --------------------------------------------------------------
# YouTube Learning Workflow (Streamlit, uses NEW pipeline)
# --------------------------------------------------------------
def render_learning_form():
    topic = st.session_state.learning_topic
    if not topic:
        return

    st.markdown(f"### 🎥 Teach me about: **{topic}**")
    st.markdown(
        "Paste a YouTube URL below so I can learn this topic and update my AWS Knowledge Base."
    )

    with st.form("yt_learn_form"):
        url = st.text_input("YouTube URL")
        submitted = st.form_submit_button("Teach me from this video")

    if submitted:
        if not url.strip():
            st.error("Please provide a valid YouTube URL.")
            return

        add_chat("user", f"Learn → {topic}\n{url}")

        # 1) Transcript
        with st.spinner("📥 Extracting transcript..."):
            path = get_youtube_transcript(url)
            if not path:
                st.error("❌ Failed to extract transcript from this URL.")
                return

        # 2) NLP preprocessing
        with st.spinner("🧠 NLP preprocessing..."):
            run_nlp_pipeline()

        # 3) Build transcript vectorstore
        with st.spinner("🏗️ Building transcript vectorstore..."):
            build_transcript_vectorstore()

        refresh_rag_agent()

        # 4) Generate explanation using RAG (hybrid retrieval: transcript + KB)
        with st.spinner("🤖 Learning from transcript (teaching myself)..."):
            explanation = RAG_Agent.teach(topic)

        # 5) Append to KB & rebuild AWS KB store
        with st.spinner("📘 Updating AWS Knowledge Base..."):
            append_to_aws_kb(topic, explanation)
            rebuild_aws_kb_vectorstore()
            load_latest_aws_kb_vectorstore()
            refresh_rag_agent()

        add_chat("assistant", f"✅ I’ve learned **{topic}** from YouTube and updated my AWS KB!")
        st.session_state.learning_topic = None
        st.rerun()


# --------------------------------------------------------------
# Initialize state
# --------------------------------------------------------------
init_state()

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☁️ AWS AI Coach")
    st.divider()

    mode_choice = st.radio(
        label="Mode",
        options=[
            "🤖 Auto (Recommended)",
            "📘 Teach / Q&A (Manual)",
            "📝 Quiz (Manual)",
            "💻 Code Helper (Manual)",
            "🎥 Learn from YouTube (Manual)"
        ]
    )

    if mode_choice.startswith("🤖"):
        st.session_state.mode_setting = "AUTO"
    else:
        st.session_state.mode_setting = "MANUAL"
        if mode_choice.startswith("📘"):
            st.session_state.mode = "TEACH"
        elif mode_choice.startswith("📝"):
            st.session_state.mode = "QUIZ"
        elif mode_choice.startswith("💻"):
            st.session_state.mode = "CODE"
        else:
            st.session_state.mode = "YOUTUBE"

    st.divider()

    if st.button("🔄 Reload latest AWS KB"):
        with st.spinner("Reloading AWS KB..."):
            load_latest_aws_kb_vectorstore()
            refresh_rag_agent()
            st.session_state.kb_loaded = True
        st.success("AWS KB reloaded!")

    if st.button("🧹 Clear chat"):
        st.session_state.chat = []
        st.session_state.pending_quiz = None
        st.session_state.quiz_answers = []
        st.session_state.quiz_topic = ""
        st.session_state.learning_topic = None
        st.session_state.mode = "TEACH"
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

if not st.session_state.kb_loaded:
    with st.spinner("Loading AWS KB..."):
        load_latest_aws_kb_vectorstore()
        refresh_rag_agent()
    st.session_state.kb_loaded = True

render_chat()

# --------------------------------------------------------------
# AUTO MODE
# --------------------------------------------------------------
if st.session_state.mode_setting == "AUTO":
    st.info("🤖 Auto mode: I’ll decide whether to explain, quiz, or generate code — and learn from YouTube when needed.")

    user_msg = st.chat_input("Ask anything about AWS...")

    if user_msg:
        add_chat("user", user_msg)
        intent = RAG_Agent.classify_intent(user_msg)

        # TEACH uses KB + relevance
        if intent == "TEACH":
            topic_norm = normalize_topic(user_msg)

            with st.spinner("Checking my AWS Knowledge Base..."):
                doc, dist, err = retrieve_from_kb(topic_norm)

            if doc is None:
                add_chat(
                    "assistant",
                    f"❌ I don't have **{topic_norm}** in my AWS KB yet.\n\n"
                    f"Please paste a YouTube URL below so I can learn it."
                )
                st.session_state.learning_topic = topic_norm
                st.rerun()
            else:
                with st.spinner("📘 Explaining from AWS KB..."):
                    answer = RAG_Agent.teach(topic_norm)
                add_chat("assistant", answer)
                st.rerun()

        # QUIZ = pure LLM, no KB gating
        elif intent == "QUIZ":
            topic = RAG_Agent.normalize_query(user_msg)
            with st.spinner("📝 Generating quiz..."):
                quiz = RAG_Agent.generate_quiz(topic)
            if quiz:
                st.session_state.pending_quiz = quiz
                st.session_state.quiz_topic = topic
                st.session_state.quiz_answers = [""] * len(quiz)
                add_chat("assistant", f"📝 Quiz ready for **{topic}** — scroll down to answer.")
            else:
                add_chat("assistant", "❌ I failed to generate a quiz for that topic.")
            st.rerun()

        # CODE = direct LLM-based helper (RAG_Agent decides context)
        else:  # CODE
            norm_q = RAG_Agent.normalize_query(user_msg)
            with st.spinner("💻 Generating AWS code help..."):
                out = RAG_Agent.generate_code_answer(norm_q)
            add_chat("assistant", out)
            st.rerun()

    # If a quiz is pending, show UI
    if st.session_state.pending_quiz:
        render_quiz_ui()

    # If a learning topic is pending, show the YouTube form
    if st.session_state.learning_topic:
        render_learning_form()

# --------------------------------------------------------------
# MANUAL MODE
# --------------------------------------------------------------
else:
    # TEACH (manual)
    if st.session_state.mode == "TEACH":
        st.info("📘 Teach / Q&A (Manual)")
        user_q = st.chat_input("Ask an AWS question...")

        if user_q:
            add_chat("user", user_q)
            topic_norm = normalize_topic(user_q)

            with st.spinner("Checking my AWS Knowledge Base..."):
                doc, dist, err = retrieve_from_kb(topic_norm)

            if doc is None:
                add_chat(
                    "assistant",
                    f"❌ I don't have **{topic_norm}** yet.\n\n"
                    f"Paste a YouTube URL below so I can learn it."
                )
                st.session_state.learning_topic = topic_norm
            else:
                with st.spinner("📘 Explaining from AWS KB..."):
                    answer = RAG_Agent.teach(topic_norm)
                add_chat("assistant", answer)
            st.rerun()

        if st.session_state.learning_topic:
            render_learning_form()

    # QUIZ (manual) — LLM-only, no KB checks
    elif st.session_state.mode == "QUIZ":
        st.warning("📝 Quiz mode (Manual)")

        if st.session_state.pending_quiz is None:
            ask = st.chat_input("Enter an AWS topic for quiz:")
            if ask:
                add_chat("user", ask)
                topic = RAG_Agent.normalize_query(ask)

                with st.spinner("📝 Generating quiz..."):
                    quiz = RAG_Agent.generate_quiz(topic)

                if quiz:
                    st.session_state.pending_quiz = quiz
                    st.session_state.quiz_topic = topic
                    st.session_state.quiz_answers = [""] * len(quiz)
                    add_chat("assistant", f"📝 Quiz ready for **{topic}** — scroll down to answer.")
                else:
                    add_chat("assistant", "❌ I failed to generate a quiz.")
                st.rerun()
        else:
            render_quiz_ui()

        if st.session_state.learning_topic:
            render_learning_form()

    # CODE (manual)
    elif st.session_state.mode == "CODE":
        st.success("💻 Code Helper (Manual)")
        ask = st.chat_input("Ask for AWS code help...")

        if ask:
            add_chat("user", ask)
            norm_q = RAG_Agent.normalize_query(ask)
            with st.spinner("Generating AWS code help..."):
                out = RAG_Agent.generate_code_answer(norm_q)
            add_chat("assistant", out)
            st.rerun()

    # Manual YOUTUBE mode
    else:
        st.markdown("### 🎥 Learn from YouTube (Manual)")

        with st.form("yt_manual_form"):
            topic = st.text_input("AWS topic to learn:")
            url = st.text_input("YouTube URL:")
            submitted = st.form_submit_button("Learn Now")

        if submitted:
            if not topic.strip() or not url.strip():
                st.error("Both topic and URL are required.")
            else:
                topic_norm = normalize_topic(topic)
                st.session_state.learning_topic = topic_norm

                add_chat("user", f"Learn → {topic_norm}\n{url}")

                with st.spinner("📥 Extracting transcript..."):
                    path = get_youtube_transcript(url)
                    if not path:
                        st.error("❌ Failed to extract transcript.")
                        st.stop()

                with st.spinner("🧠 NLP preprocessing..."):
                    run_nlp_pipeline()

                with st.spinner("🏗️ Building transcript vectorstore..."):
                    build_transcript_vectorstore()

                refresh_rag_agent()

                with st.spinner("🤖 Generating explanation..."):
                    explanation = RAG_Agent.teach(topic_norm)

                with st.spinner("📘 Updating AWS Knowledge Base..."):
                    append_to_aws_kb(topic_norm, explanation)
                    rebuild_aws_kb_vectorstore()
                    load_latest_aws_kb_vectorstore()
                    refresh_rag_agent()

                add_chat("assistant", f"✅ I’ve learned **{topic_norm}** and updated my AWS KB!")
                st.session_state.learning_topic = None
                st.rerun()
