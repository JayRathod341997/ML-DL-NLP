"""Interactive Q&A CLI for the RAG pipeline.

Usage:
    uv run python scripts/ask.py
    uv run python scripts/ask.py --question "What is attention mechanism?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline

console = Console()


def display_response(response) -> None:
    console.print(Panel(response.answer, title=f"[bold cyan]Answer[/bold cyan]", expand=False))
    console.print("\n[bold]Sources:[/bold]")
    for i, chunk in enumerate(response.context_chunks, 1):
        src = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "")
        page_str = f", page {page}" if page else ""
        console.print(f"  [{i}] {src}{page_str} (score: {chunk.score:.3f})")
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Question Answering")
    parser.add_argument("--question", help="Single question and exit")
    args = parser.parse_args()

    console.print("[bold cyan]Initialising RAG pipeline...[/bold cyan]")
    pipeline = RAGPipeline(RAGConfig())
    console.print(f"[green]Ready! {pipeline.retriever.count()} chunks indexed.[/green]\n")

    if args.question:
        response = pipeline.ask(args.question)
        display_response(response)
        return

    console.print("[bold]Document Q&A[/bold] — type 'quit' to exit\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        response = pipeline.ask(question)
        display_response(response)


if __name__ == "__main__":
    main()
