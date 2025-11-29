# --------------------------------------------------------------
# main.py (FINAL — Clean, Correct, Works with New Pipeline)
# --------------------------------------------------------------

import importlib

from transcript_extraction import get_youtube_transcript
from NLP import run_nlp_pipeline
from vector_store import build_transcript_vectorstore

from aws.aws_info import (
    retrieve_from_kb,
    append_to_aws_kb,
    rebuild_aws_kb_vectorstore,
)

import RAG_Agent


# --------------------------------------------------------------
# YouTube Learning Workflow
# --------------------------------------------------------------
def learn_from_youtube(topic: str):
    print(f"\n⚠️ I do NOT have information on '{topic}'. I need to learn it.")

    url = input("\n🔗 Enter YouTube URL (or 'q' to cancel): ").strip()
    if url.lower() == "q":
        print("\n↩️ Cancelled. Returning to main menu...")
        return

    # ---- Step 1: Get transcript ----
    print("\n📥 Extracting transcript...")
    path = get_youtube_transcript(url)
    if not path:
        print("❌ Could not extract transcript. Aborting.")
        return

    # ---- Step 2: NLP preprocessing ----
    print("\n🧠 NLP preprocessing...")
    run_nlp_pipeline()

    # ---- Step 3: Build transcript vectorstore ----
    print("\n🏗️ Building transcript vectorstore...")
    build_transcript_vectorstore()

    # Reload RAG agent so it picks up new transcript retriever
    importlib.reload(RAG_Agent)

    # ---- Step 4: Generate explanation using transcript ----
    print("\n🤖 Learning from transcript...")
    explanation = RAG_Agent.teach(topic)

    if "I don't see" in explanation.lower():
        print("❌ Could not learn enough from transcript.")
        return

    # ---- Add to knowledge base ----
    print("\n📘 Adding to AWS Knowledge Base...")
    append_to_aws_kb(topic, explanation)

    print("🔒 Rebuilding AWS KB vectorstore...")
    rebuild_aws_kb_vectorstore()

    importlib.reload(RAG_Agent)
    print("\n✅ Successfully learned new topic!")


# --------------------------------------------------------------
# MAIN MENU
# --------------------------------------------------------------
def main():
    print("🚀 Welcome to AWS AI Coach!\n")
    print("I can TEACH, QUIZ, generate CODE, or LEARN from YouTube.\n")

    importlib.reload(RAG_Agent)

    while True:
        print("""
======================================
📚 AWS AI Coach — Main Menu
======================================
1️⃣  Ask Question / Teach Topic / Quiz / Code
2️⃣  Teach me a new AWS topic from YouTube
q️⃣  Quit
======================================
""")

        choice = input("👉 Select (1/2/q): ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\n👋 Goodbye!")
            break

        # --------------------------------------------------------------
        # 1️⃣ User entered a question / quiz request / code request
        # --------------------------------------------------------------
        if choice == "1":
            user_input = input("🧠 Enter AWS question/topic/quiz/code request: ").strip()

            # Classify intent
            intent = RAG_Agent.classify_intent(user_input)
            normalized = RAG_Agent.normalize_query(user_input)

            # ---------------- TEACH MODE ----------------
            if intent == "TEACH":
                kb_doc, dist, err = retrieve_from_kb(normalized)

                if not kb_doc:
                    print(f"\n❌ Topic '{normalized}' not found in AWS KB.")
                    learn_from_youtube(normalized)
                    continue

                print("\n📘 Explanation:\n")
                answer = RAG_Agent.teach(normalized)
                print(answer)
                print("\n-----------------------------------\n")
                continue

            # ---------------- QUIZ MODE ----------------
            if intent == "QUIZ":
                print("\n📝 Generating quiz...\n")
                try:
                    quiz = RAG_Agent.generate_quiz(normalized)
                except Exception as e:
                    print(f"❌ Quiz generation failed: {e}")
                    continue

                # Print questions
                for i, q in enumerate(quiz, 1):
                    print(f"Q{i}. {q['question']}")
                    for opt, txt in q["options"].items():
                        print(f"   {opt}) {txt}")
                    print()

                # Collect answers
                responses = []
                for i in range(len(quiz)):
                    ans = input(f"Answer for Q{i+1} (A/B/C/D): ").strip().upper()
                    while ans not in ("A", "B", "C", "D"):
                        ans = input("❌ Invalid. Enter A/B/C/D: ").strip().upper()
                    responses.append(ans)

                score, report = RAG_Agent.grade_quiz(quiz, responses)
                print(f"\n🏁 FINAL SCORE: {score}/{len(quiz)}\n")

                for r in report:
                    icon = "✅" if r["ok"] else "❌"
                    print(f"{icon} Q{r['q']} — You: {r['you']} | Correct: {r['correct']}")
                    print(f"   💬 {r['explanation']}\n")

                continue

            # ---------------- CODE MODE ----------------
            if intent == "CODE":
                print("\n💻 Generating AWS code...\n")
                answer = RAG_Agent.generate_code_answer(normalized)
                print(answer)
                print("\n-----------------------------------\n")
                continue

        # --------------------------------------------------------------
        # 2️⃣ Manual YouTube learning mode
        # --------------------------------------------------------------
        elif choice == "2":
            topic = input("🧠 Enter topic name: ").strip()
            learn_from_youtube(topic)

        else:
            print("❌ Invalid input. Choose 1, 2, or q.\n")


# --------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Goodbye.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
