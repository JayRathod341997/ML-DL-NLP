from __future__ import annotations


class Postprocessor:
    """Merges model predictions back with input rows."""

    def merge(
        self,
        rows: list[dict],
        predictions: list[dict],
    ) -> list[dict]:
        """Merge input rows with model predictions.

        Args:
            rows: Input rows from the data reader.
            predictions: List of {"label": str, "score": float} dicts.

        Returns:
            List of merged result dicts with text, label, score, row_index.
        """
        results = []
        for row, pred in zip(rows, predictions):
            results.append({
                "row_index": row.get("row_index", -1),
                "text": row.get("text", ""),
                "label": pred["label"],
                "score": pred["score"],
            })
        return results
