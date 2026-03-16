"""Train a LambdaMART model on MS MARCO.

Usage:
    uv run python scripts/train_lambdamart.py --max-queries 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.data_processor import load_msmarco_examples, examples_to_feature_matrix
from src.lambdamart_model import LambdaMARTModel
from src.metrics import evaluate_ranking


def compute_bm25_scores(examples):
    """Pre-compute BM25 scores for all (query, doc) pairs grouped by query."""
    from collections import defaultdict
    query_docs = defaultdict(list)
    for ex in examples:
        query_docs[ex.query_id].append(ex)

    bm25_scores = {}
    for qid, query_examples in tqdm(query_docs.items(), desc="BM25 scoring"):
        docs_tokenised = [e.text.lower().split() for e in query_examples]
        bm25 = BM25Okapi(docs_tokenised)
        q_tokens = query_examples[0].query.lower().split()
        scores = bm25.get_scores(q_tokens)
        for ex, score in zip(query_examples, scores):
            bm25_scores[(ex.query_id, ex.doc_id)] = float(score)
    return bm25_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("models/lambdamart.pkl"))
    args = parser.parse_args()

    print(f"Loading {args.max_queries} MS MARCO queries...")
    examples = load_msmarco_examples(max_queries=args.max_queries)
    print(f"Loaded {len(examples)} (query, doc) pairs")

    print("Computing BM25 features...")
    bm25_scores = compute_bm25_scores(examples)

    X, y, groups = examples_to_feature_matrix(examples, bm25_scores)
    print(f"Feature matrix: {X.shape}, groups: {len(groups)}")

    print("Training LambdaMART...")
    model = LambdaMARTModel()
    model.fit(X, y, groups)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
