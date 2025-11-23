# --------------------------------------------------------------
# main.py  (FINAL — fixed QUIZ routing)
# --------------------------------------------------------------

import importlib

from transcript_extraction import get_youtube_transcript
from NLP import run_nlp_pipeline
from vector_store import build_vector_store

from aws.aws_info import (
    is_topic_in_aws_kb,
    append_to_aws_kb,
    rebuild_aws_kb_vectorstore,
    load_latest_vectorstore,
)

import RAG_Agent


# --------------------------------------------------------------
# YouTube Learning Flow
# --------------------------------------------------------------
def learn_from_youtube(topic: str):
    print(f"\n⚠️ I don't have info on '{topic}'. Let's learn it!")

    url = input("🔗 Enter YouTube URL OR 'q' to cancel: ").strip()
    if url.lower() == "q":
        print("\n↩️ Cancelled. Returning to main menu...\n")
        return

    # 1) Transcript
    print("\n📥 Extracting transcript...")
    get_youtube_transcript(url)

    # 2) NLP
    print("\n🧠 NLP preprocessing...")
    run_nlp_pipeline()

    # 3) Transcript vectorstore
    print("\n🏗️ Building transcript VectorStore for RAG...")
    build_vector_store()

    # reload transcript logic
    importlib.reload(RAG_Agent)

    # 4) Learn from transcript RAG
    print("\n🤖 Learning from transcript and generating explanation...\n")
    explanation = RAG_Agent.teach(topic)

    print("\n📘 Adding this explanation into AWS Knowledge Base...")
    append_to_aws_kb(topic, explanation)

    print("\n🔒 Creating NEW KB VectorStore version...")
    rebuild_aws_kb_vectorstore()

    load_latest_vectorstore()
    importlib.reload(RAG_Agent)

    print("\n✅ Learned new topic and updated AWS KB!")
    print("\n🎉 You can now ask me about this topic directly.\n")


# --------------------------------------------------------------
# Main Menu
# --------------------------------------------------------------
def main():
    print("🚀 Welcome to AWS AI Coach!")
    print("I can teach AWS topics, quiz you, generate code, or learn from YouTube.\n")

    load_latest_vectorstore()
    importlib.reload(RAG_Agent)

    while True:
        print("""
======================================
📚 AWS AI Coach - Main Menu
======================================
1️⃣  Train on an AWS Topic
    (Explain topics, answer questions, quizzes, code helper)
2️⃣  Provide YouTube URL (force learning)
q️⃣  Quit
======================================
""")

        choice = input("👉 Select (1/2/q): ").strip().lower()

        if choice in ["q", "quit", "exit"]:
            print("\n👋 Goodbye!")
            break

        # ----------------------------------------------------------
        # 1️⃣ TRAIN / QUIZ / CODE / EXPLAIN
        # ----------------------------------------------------------
        if choice == "1":
            raw_input_query = input("🧠 Enter topic/question/quiz/code request: ").strip()

            # Detect intent BEFORE anything else
            intent = RAG_Agent.classify_intent(raw_input_query)

            # Normalize the query
            norm_query = RAG_Agent.normalize_query(raw_input_query)

            # Always load latest KB
            load_latest_vectorstore()

            # TEACH mode → need to check KB relevance
            if intent == "TEACH":
                if not is_topic_in_aws_kb(norm_query):
                    learn_from_youtube(norm_query)
                    continue
                RAG_Agent.teach(norm_query)
                continue

            # QUIZ mode
            if intent == "QUIZ":
                quiz = RAG_Agent.generate_quiz(norm_query)
                if not quiz:
                    print("❌ No quiz available. Provide YouTube URL or refine your query.")
                    continue

                # Print MCQs
                RAG_Agent.print_quiz(quiz)
                print("\n💡 Enter your answers (A/B/C/D):")

                answers = []
                for i in range(len(quiz)):
                    ans = input(f"Answer for Q{i+1}: ").strip().upper()
                    while ans not in ["A", "B", "C", "D"]:
                        ans = input("Please enter A/B/C/D: ").strip().upper()
                    answers.append(ans)

                # Score the quiz
                score, report = RAG_Agent.grade_quiz(quiz, answers)

                print(f"\n🏁 Your Final Score: {score}/{len(quiz)}")
                print("\n📘 Detailed Feedback:\n")

                for r in report:
                    icon = "✅" if r["ok"] else "❌"
                    print(f"{icon} Q{r['q']} — You: {r['you']} | Correct: {r['correct']}")
                    print(f"   💬 {r['explanation']}\n")
                continue

            # CODE mode
            if intent == "CODE":
                RAG_Agent.generate_code_answer(norm_query)
                continue

        # ----------------------------------------------------------
        # 2️⃣ MANUAL YOUTUBE LEARNING
        # ----------------------------------------------------------
        elif choice == "2":
            topic = input("🧠 Name the topic you want me to learn: ").strip()
            learn_from_youtube(topic)

        else:
            print("❌ Invalid option. Please choose 1, 2, or q.\n")


# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
