from __future__ import annotations

import sys

from dotenv import load_dotenv

from src.config import load_settings
from src.pipeline.rag_pipeline import RAGPipeline


QUESTION = "What is the warranty period?"
SESSION_ID = "qa-test-sarvam"
PREFERRED_LANGUAGE = None
SAMPLE_QUERIES = [
    "What is the warranty period?",
    "Do you provide training after purchase?",
    "Which machine model is best for advanced woodworking?",
    "What materials can your CNC machines process?",
    "Is EMI or finance available?",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()
    settings = load_settings()
    pipeline = RAGPipeline(settings)
    pipeline.initialize()

    response = pipeline.query(
        question=QUESTION,
        session_id=SESSION_ID,
        preferred_language=PREFERRED_LANGUAGE,
    )

    print(f"Question: {QUESTION}")
    print()
    print("Answer:")
    print(response["answer"])
    print()
    print("Sources:")
    for source in response.get("sources", []):
        print(f"- {source}")
    print()
    print("Retrieval:")
    for item in response.get("retrieval", []):
        faq_index = item.get("metadata", {}).get("faq_index")
        score = item.get("score")
        question_text = item.get("metadata", {}).get("question", "")
        print(f"- faq={faq_index} score={score} question={question_text}")
    print()
    print("Sample queries you can try:")
    for sample in SAMPLE_QUERIES:
        print(f"- {sample}")


if __name__ == "__main__":
    main()
