from __future__ import annotations

from dotenv import load_dotenv

from src.config import load_settings
from src.pipeline.rag_pipepline import RAGPipeline


def run_cli() -> None:
	load_dotenv()
	settings = load_settings()
	rag = RAGPipeline(settings)
	rag.initialize(force_rebuild=False)

	print("RAG system initialized. Type 'exit' to quit.")
	session_id = "cli-session"

	while True:
		question = input("\nYou: ").strip()
		if not question:
			continue
		if question.lower() in {"exit", "quit"}:
			break

		response = rag.query(question=question, session_id=session_id)
		print(f"\nAssistant: {response['answer']}")
		if response["sources"]:
			print("Sources:")
			for source in response["sources"]:
				print(f"- {source}")
		if response["images"]:
			print("Images:")
			for image in response["images"]:
				print(f"- {image}")


if __name__ == "__main__":
	run_cli()

