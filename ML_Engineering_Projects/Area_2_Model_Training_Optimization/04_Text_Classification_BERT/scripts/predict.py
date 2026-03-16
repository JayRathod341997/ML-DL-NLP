"""Run inference on new texts using a trained checkpoint.

Usage:
    uv run python scripts/predict.py --text "Scientists discover new dinosaur species"
    uv run python scripts/predict.py --file texts.txt --checkpoint checkpoints/best_model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainConfig
from src.predict import Predictor

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict text class")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--file", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model"))
    args = parser.parse_args()

    config = TrainConfig()
    predictor = Predictor(str(args.checkpoint), config)

    if args.text:
        label_idx, confidence = predictor.predict(args.text)
        label = AG_NEWS_LABELS.get(label_idx, str(label_idx))
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
    else:
        texts = args.file.read_text().strip().split("\n")
        for text in texts:
            if text.strip():
                label_idx, confidence = predictor.predict(text.strip())
                label = AG_NEWS_LABELS.get(label_idx, str(label_idx))
                print(f"[{label} {confidence:.3f}] {text[:80]}")


if __name__ == "__main__":
    main()
