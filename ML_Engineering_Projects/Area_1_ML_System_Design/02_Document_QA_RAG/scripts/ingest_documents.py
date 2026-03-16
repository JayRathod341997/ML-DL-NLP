"""Ingest documents into the RAG vector store.

Usage:
    uv run python scripts/ingest_documents.py --dir data/pdfs/
    uv run python scripts/ingest_documents.py --file data/pdfs/paper.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track

from src.config import RAGConfig
from src.document_loader import DocumentLoader
from src.chunker import RecursiveChunker
from src.retriever import ChromaRetriever

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Single file to ingest")
    group.add_argument("--dir", type=Path, help="Directory of documents to ingest")
    args = parser.parse_args()

    config = RAGConfig()
    loader = DocumentLoader()
    chunker = RecursiveChunker(config)
    retriever = ChromaRetriever(config)

    if args.file:
        console.print(f"[cyan]Loading {args.file}...[/cyan]")
        docs = loader.load(args.file)
    else:
        console.print(f"[cyan]Loading documents from {args.dir}...[/cyan]")
        docs = loader.load_directory(args.dir)

    console.print(f"[green]Loaded {len(docs)} document pages[/green]")

    chunks = chunker.chunk_all(docs)
    console.print(f"[green]Created {len(chunks)} chunks[/green]")

    console.print("[bold]Adding to vector store...[/bold]")
    retriever.add_chunks(chunks)

    console.print(f"[bold green]Done! Vector store now has {retriever.count()} chunks.[/bold green]")


if __name__ == "__main__":
    main()
