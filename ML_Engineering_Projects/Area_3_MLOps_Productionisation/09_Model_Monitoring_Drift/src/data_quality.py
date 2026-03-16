from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataQualityReport:
    num_rows: int
    missing_text_count: int
    missing_text_pct: float
    empty_text_count: int
    avg_text_length: float
    min_text_length: int
    max_text_length: int
    very_short_count: int   # texts < 10 chars
    very_long_count: int    # texts > 5000 chars
    issues: list[str]       # human-readable issue descriptions


class DataQualityChecker:
    """Checks basic data quality issues for text datasets."""

    def check(self, df: pd.DataFrame, text_column: str = "text") -> DataQualityReport:
        total = len(df)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")

        texts = df[text_column]
        missing = texts.isna().sum()
        non_null = texts.dropna().astype(str)
        empty = (non_null.str.strip() == "").sum()
        lengths = non_null.str.len()
        avg_len = float(lengths.mean()) if len(lengths) > 0 else 0
        very_short = (lengths < 10).sum()
        very_long = (lengths > 5000).sum()

        issues = []
        if missing / total > 0.01:
            issues.append(f"HIGH MISSING RATE: {missing/total:.1%} of texts are null")
        if empty / total > 0.01:
            issues.append(f"HIGH EMPTY RATE: {empty/total:.1%} of texts are empty")
        if very_short / total > 0.05:
            issues.append(f"MANY SHORT TEXTS: {very_short/total:.1%} texts < 10 chars")
        if very_long / total > 0.05:
            issues.append(f"MANY LONG TEXTS: {very_long/total:.1%} texts > 5000 chars")

        return DataQualityReport(
            num_rows=total,
            missing_text_count=int(missing),
            missing_text_pct=round(float(missing / total), 4),
            empty_text_count=int(empty),
            avg_text_length=round(avg_len, 1),
            min_text_length=int(lengths.min()) if len(lengths) > 0 else 0,
            max_text_length=int(lengths.max()) if len(lengths) > 0 else 0,
            very_short_count=int(very_short),
            very_long_count=int(very_long),
            issues=issues,
        )
