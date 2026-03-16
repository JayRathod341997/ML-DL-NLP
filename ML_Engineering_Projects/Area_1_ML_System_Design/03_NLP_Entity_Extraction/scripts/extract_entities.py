"""Extract named entities from a document file.

Usage:
    uv run python scripts/extract_entities.py --file report.pdf
    uv run python scripts/extract_entities.py --file report.pdf --output results.json
    uv run python scripts/extract_entities.py --text "Apple was founded by Steve Jobs in Cupertino."
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.ner_model import NERModel
from src.document_parser import DocumentParser
from src.entity_aggregator import EntityAggregator
from src.relation_extractor import RelationExtractor
from src.output_formatter import OutputFormatter

console = Console()


def run_on_text(text: str, model: NERModel) -> None:
    entities_raw = model.predict(text)
    aggregator = EntityAggregator()
    entities = aggregator.aggregate(entities_raw)
    table = Table(title="Extracted Entities", show_lines=True)
    table.add_column("Entity", style="bold")
    table.add_column("Label", style="cyan")
    table.add_column("Count")
    table.add_column("Confidence")
    for e in entities:
        table.add_row(e.text, e.label, str(e.count), f"{e.confidence:.3f}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP Entity Extraction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path)
    group.add_argument("--text", type=str)
    parser.add_argument("--output", type=Path, help="Save JSON output to this path")
    parser.add_argument("--model", default="dslim/bert-base-NER")
    args = parser.parse_args()

    console.print(f"[cyan]Loading NER model: {args.model}...[/cyan]")
    model = NERModel(hf_model=args.model)

    if args.text:
        run_on_text(args.text, model)
        return

    console.print(f"[cyan]Parsing {args.file}...[/cyan]")
    doc_parser = DocumentParser()
    pages = doc_parser.parse(args.file)
    full_text = "\n\n".join(p.text for p in pages)

    console.print(f"[green]Parsed {len(pages)} pages[/green]")
    console.print("[bold]Extracting entities...[/bold]")

    all_entities = []
    for page in pages:
        all_entities.extend(model.predict(page.text))

    aggregator = EntityAggregator()
    entities = aggregator.aggregate(all_entities)

    extractor = RelationExtractor()
    relations = extractor.extract_from_text(full_text, entities)

    formatter = OutputFormatter()
    result = formatter.to_json(
        source=args.file.name,
        entities=entities,
        relations=relations,
        output_path=args.output,
    )

    if args.output:
        console.print(f"[bold green]Saved to {args.output}[/bold green]")
    else:
        console.print_json(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
