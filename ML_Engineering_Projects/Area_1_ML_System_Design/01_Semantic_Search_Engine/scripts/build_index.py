"""Build and persist a semantic search index from a HuggingFace dataset.

Usage:
    uv run python scripts/build_index.py --dataset ag_news
    uv run python scripts/build_index.py --dataset wikipedia --streaming
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from rich.console import Console

from src.config import SearchConfig
from src.indexer import Document, DocumentIndexer

console = Console()


def load_ag_news(streaming: bool = False) -> list[Document]:
    console.print("[cyan]Loading AG News dataset...[/cyan]")
    ds = load_dataset("ag_news", split="train", streaming=streaming)
    docs = []
    for i, row in enumerate(ds):
        docs.append(
            Document(
                doc_id=f"ag_{i}",
                text=row["text"],
                metadata={"label": row["label"], "source": "ag_news"},
            )
        )
    console.print(f"[green]Loaded {len(docs)} documents[/green]")
    return docs


def load_wikipedia(streaming: bool = True, max_docs: int = 50_000) -> list[Document]:
    console.print("[cyan]Loading Wikipedia (simple) dataset...[/cyan]")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.simple", split="train", streaming=streaming
    )
    docs = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        text = row["text"][:1000]  # truncate to first 1000 chars
        docs.append(
            Document(
                doc_id=f"wiki_{i}",
                text=text,
                metadata={"title": row["title"], "source": "wikipedia"},
            )
        )
        if (i + 1) % 5000 == 0:
            console.print(f"  Loaded {i + 1} docs...")
    console.print(f"[green]Loaded {len(docs)} documents[/green]")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantic search index")
    parser.add_argument(
        "--dataset", choices=["ag_news", "wikipedia"], default="ag_news"
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming mode (saves memory)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=50_000, help="Max docs for Wikipedia"
    )
    args = parser.parse_args()

    config = SearchConfig()
    console.print(f"[bold]Config:[/bold] model={config.embed_model}, store={config.vector_store}")

    if args.dataset == "ag_news":
        docs = load_ag_news(streaming=args.streaming)
    else:
        docs = load_wikipedia(streaming=args.streaming, max_docs=args.max_docs)

    console.print(f"[bold]Building index with {len(docs)} documents...[/bold]")
    indexer = DocumentIndexer(config)
    indexer.index_batched(docs)
    indexer.save()
    console.print(f"[bold green]Index saved! Total docs: {len(indexer)}[/bold green]")


if __name__ == "__main__":
    main()
