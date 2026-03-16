"""Batch process a directory of documents for entity extraction.

Usage:
    uv run python scripts/batch_process.py --dir data/documents/ --output-dir results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from rich.console import Console

from src.ner_model import NERModel
from src.document_parser import DocumentParser
from src.entity_aggregator import EntityAggregator
from src.relation_extractor import RelationExtractor
from src.output_formatter import OutputFormatter

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch entity extraction")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model", default="dslim/bert-base-NER")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_parser = DocumentParser()
    model = NERModel(hf_model=args.model)
    aggregator = EntityAggregator()
    rel_extractor = RelationExtractor()
    formatter = OutputFormatter()

    supported = [".pdf", ".docx", ".txt", ".md"]
    files = [f for ext in supported for f in args.dir.rglob(f"*{ext}")]
    console.print(f"[green]Found {len(files)} files to process[/green]")

    for file in tqdm(files, desc="Processing"):
        try:
            pages = doc_parser.parse(file)
            full_text = "\n\n".join(p.text for p in pages)
            raw_entities = []
            for page in pages:
                raw_entities.extend(model.predict(page.text))
            entities = aggregator.aggregate(raw_entities)
            relations = rel_extractor.extract_from_text(full_text, entities)
            out_path = args.output_dir / (file.stem + "_entities.json")
            formatter.to_json(file.name, entities, relations, out_path)
        except Exception as e:
            console.print(f"[red]Failed {file.name}: {e}[/red]")

    console.print(f"[bold green]Done! Results in {args.output_dir}[/bold green]")


if __name__ == "__main__":
    main()
