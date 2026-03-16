"""Interactive semantic search REPL.

Usage:
    uv run python scripts/run_search.py
    uv run python scripts/run_search.py --query "machine learning optimization"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.config import SearchConfig
from src.searcher import SemanticSearcher

console = Console()


def display_results(results, query: str) -> None:
    table = Table(title=f'Results for: "{query}"', show_lines=True)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Text", style="white")
    table.add_column("Metadata", style="dim")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.text[:200] + ("..." if len(r.text) > 200 else ""),
            str(r.metadata),
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive semantic search")
    parser.add_argument("--query", help="Run a single query and exit")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    console.print("[bold cyan]Loading index...[/bold cyan]")
    config = SearchConfig(top_k=args.top_k)
    searcher = SemanticSearcher(config)
    console.print(f"[green]Index loaded ({len(searcher.store)} documents)[/green]")

    if args.query:
        results = searcher.search(args.query)
        display_results(results, args.query)
        return

    console.print("[bold]Semantic Search REPL[/bold] — type 'quit' to exit\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        results = searcher.search(query)
        display_results(results, query)


if __name__ == "__main__":
    main()
