from __future__ import annotations

from sentence_transformers import InputExample


def load_nli_pairs(
    max_samples: int | None = None,
    include_multinli: bool = True,
) -> list[InputExample]:
    """Load NLI entailment pairs as (anchor, positive) for MNRL training.

    Filters to only entailment pairs (label=0 in SNLI/MultiNLI).
    """
    from datasets import load_dataset

    examples = []

    # SNLI
    ds = load_dataset("stanfordnlp/snli", split="train")
    for row in ds:
        if row["label"] == 0:  # entailment
            examples.append(InputExample(texts=[row["premise"], row["hypothesis"]]))
        if max_samples and len(examples) >= max_samples // 2:
            break

    # MultiNLI
    if include_multinli:
        mnli = load_dataset("nyu-mll/multi_nli", split="train")
        for row in mnli:
            if row["label"] == 0:
                examples.append(InputExample(texts=[row["premise"], row["hypothesis"]]))
            if max_samples and len(examples) >= max_samples:
                break

    print(f"Loaded {len(examples)} NLI entailment pairs")
    return examples


def load_stsb_pairs(split: str = "train") -> list[InputExample]:
    """Load STS Benchmark pairs with float similarity scores [0, 1].

    Normalises scores from 0-5 range to 0-1 for CosineSimilarityLoss.
    """
    from datasets import load_dataset

    ds = load_dataset("sentence-transformers/stsb", split=split)
    examples = []
    for row in ds:
        score = float(row["score"]) / 5.0  # normalise to [0, 1]
        examples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
    print(f"Loaded {len(examples)} STS-B pairs (split={split})")
    return examples
