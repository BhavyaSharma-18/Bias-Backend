"""
engine/mitigation.py
Applies AIF360 in-processing bias mitigation via Reweighing.
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

TEMP_PATH = Path("temp_dataset.csv")
FIXED_PATH = Path("fixed_dataset.csv")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every object / category column to its integer code."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    return df


def apply_reweighing(outcome_col: str, protected_attr: str) -> Dict[str, str]:
    """
    Apply the AIF360 Reweighing pre-processing algorithm to the stored
    temporary dataset and persist the transformed dataset as fixed_dataset.csv.

    Parameters
    ----------
    outcome_col : str
        Name of the binary label / outcome column.
    protected_attr : str
        Name of the protected attribute column.

    Returns
    -------
    dict
        {"status": "success", "message": "Reweighed dataset saved"}
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

    # Drop nulls in key columns
    df = df.dropna(subset=[outcome_col, protected_attr])

    if df.empty:
        raise ValueError(
            "Dataset is empty after dropping nulls in the selected columns."
        )

    # ------------------------------------------------------------------ #
    # 2. Encode categoricals
    # ------------------------------------------------------------------ #
    df = _encode_categoricals(df)

    # ------------------------------------------------------------------ #
    # 3. Build AIF360 BinaryLabelDataset
    # ------------------------------------------------------------------ #
    unprivileged_groups = [{protected_attr: 0}]
    privileged_groups = [{protected_attr: 1}]

    aif_ds = BinaryLabelDataset(
        df=df,
        label_names=[outcome_col],
        protected_attribute_names=[protected_attr],
    )

    # ------------------------------------------------------------------ #
    # 4. Apply Reweighing
    # ------------------------------------------------------------------ #
    rw = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    rw.fit(aif_ds)
    transformed_ds = rw.transform(aif_ds)

    # ------------------------------------------------------------------ #
    # 5. Convert back to DataFrame and save
    # ------------------------------------------------------------------ #
    transformed_df, _ = transformed_ds.convert_to_dataframe()
    transformed_df.to_csv(FIXED_PATH, index=False)

    return {"status": "success", "message": "Reweighed dataset saved"}
