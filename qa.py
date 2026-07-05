"""
qa.py — LiteLLM + NVIDIA NIM Multi-Turn Conversational Smoke Test.

Simulates a sequential customer chat session using the RAGPipeline.
Clears memory at start to ensure reproducibility, then executes multiple
conversational turns to verify history context preservation and RAG retrieval.

Usage:
    python qa.py
"""
from __future__ import annotations

import sys
from dotenv import load_dotenv
from src.config import load_settings
from src.pipeline.rag_pipeline import RAGPipeline

# ─── Multi-Turn Test Script ───────────────────────────────────────────────────
# We simulate a Bengali/Banglish customer asking a sequence of related questions.
TEST_SESSION_ID = "qa-multi-turn-test"
PREFERRED_LANGUAGE = "Bengali"

CONVERSATION_TURNS = [
    {
        "turn": 1,
        "question": "hello apnader kache ki ki macine ache?",
        "description": "Initial greeting & request for available machines."
    },
    {
        "turn": 2,
        "question": "apnader 1325 model er price range koto?",
        "description": "Follow-up asking for the price of a specific model mentioned in Turn 1."
    },
    {
        "turn": 3,
        "question": "eta kinte ki kono bank finance pawa jabe?",
        "description": "Context-dependent follow-up (using 'eta' / 'this') asking about financing."
    }
]
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()
    settings = load_settings()

    print("=" * 80)
    print(" LITELLM + NVIDIA NIM MULTI-TURN CONVERSATION TEST")
    print("=" * 80)
    print(f"[config] LLM Model   : {settings.llm_model}")
    print(f"[config] Temperature : {settings.llm_temperature}")
    print(f"[config] Max Tokens  : {settings.llm_max_output_tokens}")
    print(f"[config] Session ID  : {TEST_SESSION_ID}")
    print(f"[config] Language    : {PREFERRED_LANGUAGE}")
    print("=" * 80)

    # Initialize RAG
    pipeline = RAGPipeline(settings)
    pipeline.initialize()

    # Clear prior session state for clean, reproducible test runs
    pipeline.memory_manager.clear(TEST_SESSION_ID)
    print("[system] Cleared any existing session history for test run.\n")

    # Run each conversational turn
    for turn_info in CONVERSATION_TURNS:
        turn = turn_info["turn"]
        q_text = turn_info["question"]
        desc = turn_info["description"]

        print(f"--- TURN {turn} : {desc} ---")
        print(f"User Question: '{q_text}'")

        # Query RAG pipeline (this stores history inside the session store)
        response = pipeline.query(
            question=q_text,
            session_id=TEST_SESSION_ID,
            preferred_language=PREFERRED_LANGUAGE,
        )

        print("\nAnswer:")
        print(response["answer"])
        print()

        # Display what was retrieved
        print("Retrieved Chunks:")
        for idx, item in enumerate(response.get("retrieval", []), 1):
            faq_index = item.get("metadata", {}).get("faq_index")
            score = item.get("score")
            question_text = item.get("metadata", {}).get("question", "")
            print(f"  {idx}. [FAQ Q{faq_index}] (Score: {score:.4f}) -> {question_text}")

        # Show context history window
        history_window = pipeline.memory_manager.get_context_window(
            session_id=TEST_SESSION_ID,
            question=q_text,
            max_messages=6
        )
        print("\nSession History Window (Last 6 messages stored):")
        for msg in history_window:
            role = msg["role"].upper()
            content_preview = msg["content"].replace("\n", " ")[:80]
            print(f"  [{role}]: {content_preview}...")

        print("-" * 80)
        print()

    print("=" * 80)
    print(" MULTI-TURN TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()