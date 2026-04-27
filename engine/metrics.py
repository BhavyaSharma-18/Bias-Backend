"""
engine/metrics.py
Core bias-audit logic using AIF360 BinaryLabelDataset.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

TEMP_PATH = Path("temp_dataset.csv")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every object / category column to its integer code."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df


def get_verdict(dir_score: float) -> str:
    """
    Classify a disparate-impact score into a traffic-light verdict.

    GREEN  → score >= 0.9   (low risk)
    YELLOW → 0.8 <= score < 0.9  (moderate risk)
    RED    → score < 0.8   (high risk)
    """
    if dir_score < 0.8:
        return "RED"
    if dir_score < 0.9:
        return "YELLOW"
    return "GREEN"


# --------------------------------------------------------------------------- #
# Main audit function
# --------------------------------------------------------------------------- #

def run_bias_audit(outcome_col: str, protected_attr: str) -> Dict[str, Any]:
    """
    Perform a full bias audit on the stored temporary dataset.

    Parameters
    ----------
    outcome_col : str
        Name of the binary label / outcome column.
    protected_attr : str
        Name of the protected attribute column (will be binarised to 0/1).

    Returns
    -------
    dict with keys:
        disparate_impact, statistical_parity_difference,
        group_representation, proxy_variables, verdict
    """
    # ------------------------------------------------------------------ #
    # 1. Load & validate
    # ------------------------------------------------------------------ #
    if not TEMP_PATH.exists():
        raise FileNotFoundError(
            "No dataset found. Please upload a file via /api/upload first."
        )

    df = pd.read_csv(TEMP_PATH)

    for col in (outcome_col, protected_attr):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    # ------------------------------------------------------------------ #
    # 2. Drop rows with nulls in the key columns
    # ------------------------------------------------------------------ #
    df = df.dropna(subset=[outcome_col, protected_attr])

    if df.empty:
        raise ValueError(
            "Dataset is empty after dropping nulls in the selected columns."
        )

    # ------------------------------------------------------------------ #
    # 3. Encode categoricals
    # ------------------------------------------------------------------ #
    df = _encode_categoricals(df)

    # ------------------------------------------------------------------ #
    # 4. Build AIF360 BinaryLabelDataset
    # ------------------------------------------------------------------ #
    aif_ds = BinaryLabelDataset(
        df=df,
        label_names=[outcome_col],
        protected_attribute_names=[protected_attr],
    )

    unprivileged_groups: List[Dict[str, int]] = [{protected_attr: 0}]
    privileged_groups: List[Dict[str, int]] = [{protected_attr: 1}]

    # ------------------------------------------------------------------ #
    # 5. Compute fairness metrics
    # ------------------------------------------------------------------ #
    metric = BinaryLabelDatasetMetric(
        aif_ds,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    di_score: float = round(float(metric.disparate_impact()), 3)
    spd_score: float = round(float(metric.statistical_parity_difference()), 3)

    # ------------------------------------------------------------------ #
    # 6. Proxy variable detection
    # ------------------------------------------------------------------ #
    proxy_variables: List[Dict[str, Any]] = []
    feature_cols = [c for c in df.columns if c not in (outcome_col, protected_attr)]

    for col in feature_cols:
        try:
            corr = abs(df[[col, protected_attr]].corr().iloc[0, 1])
            if not np.isnan(corr) and corr > 0.6:
                proxy_variables.append(
                    {"column": col, "correlation": round(float(corr), 3)}
                )
        except Exception:
            # Skip columns that can't be correlated (e.g., all-null after encode)
            pass

    # ------------------------------------------------------------------ #
    # 7. Group representation
    # ------------------------------------------------------------------ #
    group_rep = (
        df[protected_attr]
        .value_counts(normalize=True)
        .round(3)
        .to_dict()
    )
    # JSON keys must be strings
    group_representation = {str(k): v for k, v in group_rep.items()}

    # ------------------------------------------------------------------ #
    # 8. Verdict
    # ------------------------------------------------------------------ #
    verdict = get_verdict(di_score)

    return {
        "disparate_impact": di_score,
        "statistical_parity_difference": spd_score,
        "group_representation": group_representation,
        "proxy_variables": proxy_variables,
        "verdict": verdict,
    }
