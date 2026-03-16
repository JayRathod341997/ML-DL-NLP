"""Evaluate ranking models on MS MARCO dev set.

Usage:
    uv run python scripts/evaluate_ranking.py --model bm25
    uv run python scripts/evaluate_ranking.py --model crossencoder
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from src.data_processor import load_msmarco_examples
from src.metrics import evaluate_ranking
from src.neural_reranker import CrossEncoderReranker
from src.pipeline import RankingPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bm25", "crossencoder"], default="bm25")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    console.print(f"[cyan]Loading {args.max_queries} eval queries...[/cyan]")
    examples = load_msmarco_examples(max_queries=args.max_queries, split="validation")

    # Group by query
    query_groups = defaultdict(list)
    for ex in examples:
        query_groups[ex.query_id].append(ex)

    reranker = None
    if args.model == "crossencoder":
        console.print("[cyan]Loading CrossEncoder...[/cyan]")
        reranker = CrossEncoderReranker()

    all_relevance = []
    for qid, group_examples in list(query_groups.items()):
        docs = [e.text for e in group_examples]
        relevance_map = {e.text: e.relevance for e in group_examples}
        query = group_examples[0].query

        pipeline = RankingPipeline(docs, reranker=reranker, final_top_k=args.top_k)
        results = pipeline.search(query)

        ranked_relevance = [relevance_map.get(r.doc_text, 0) for r in results]
        all_relevance.append(ranked_relevance)

    metrics = evaluate_ranking(all_relevance, k=args.top_k)
    table = Table(title=f"Ranking Results — {args.model.upper()}")
    table.add_column("Metric", style="bold")
    table.add_column("Score", style="cyan")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    main()
