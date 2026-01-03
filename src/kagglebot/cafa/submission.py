from __future__ import annotations

from math import floor, log10
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def round_to_n_significant_figures(x: float, n: int = 3) -> float:
    """Round to n significant figures."""
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def create_submission(
    predictions: np.ndarray,
    scores: np.ndarray,
    protein_ids: list[str],
    idx_to_term: dict[int, str],
    output_path: str | Path,
) -> None:
    """
    Create submission file in CAFA format.

    Format:
    protein_id\tGO:xxxxxxx\tscore
    """
    rows: list[dict[str, object]] = []

    for i, protein_id in enumerate(tqdm(protein_ids, desc="Creating submission")):
        term_indices = np.where(predictions[i] == 1)[0]

        for term_idx in term_indices:
            go_term = idx_to_term[int(term_idx)]
            score = float(scores[i, term_idx])
            if score <= 0:
                continue
            score = round_to_n_significant_figures(score, 3)
            score = min(max(score, 1e-10), 1.0)
            rows.append({"protein_id": protein_id, "go_term": go_term, "score": score})

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", header=False, index=False)

    print(f"Submission saved: {output_path}")
    print(f"Total predictions: {len(rows)}")
    if not df.empty:
        print(f"Unique proteins: {df['protein_id'].nunique()}")


def validate_submission(submission_path: str | Path, go_terms_valid: set[str]) -> bool:
    """Validate submission format."""
    submission_path = Path(submission_path)
    df = pd.read_csv(submission_path, sep="\t", header=None, names=["protein_id", "term", "score"])

    if df.empty:
        raise AssertionError("Submission file is empty")

    if not (df["score"] > 0).all():
        raise AssertionError("Found zero or negative scores")
    if not (df["score"] <= 1).all():
        raise AssertionError("Found scores > 1")

    invalid_terms = set(df["term"]) - go_terms_valid
    if invalid_terms:
        print(f"Warning: {len(invalid_terms)} invalid GO terms found")

    terms_per_protein = df.groupby("protein_id").size()
    violations = terms_per_protein[terms_per_protein > 1500]
    if len(violations) > 0:
        raise AssertionError(f"{len(violations)} proteins exceed 1500 terms")

    print("Validation passed!")
    return True
