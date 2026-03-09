from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "isFraud"
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
LEAKAGE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

RAW_FEATURE_DESCRIPTIONS = {
    "step": "Hour index in the simulation timeline.",
    "type": "Transaction type such as CASH_IN, CASH_OUT, PAYMENT, TRANSFER, or DEBIT.",
    "amount": "Transaction amount.",
    "nameOrig": "Origin account identifier.",
    "nameDest": "Destination account identifier.",
    "oldbalanceOrg": "Origin account balance before the transaction.",
    "newbalanceOrig": "Origin account balance after the transaction.",
    "oldbalanceDest": "Destination account balance before the transaction.",
    "newbalanceDest": "Destination account balance after the transaction.",
    "isFraud": "Fraud label where 1 indicates fraud and 0 indicates non-fraud.",
    "isFlaggedFraud": "Rule-engine flag raised for transfers above a high amount threshold.",
}

ENGINEERED_FEATURE_DESCRIPTIONS = {
    "hour_of_day": "Hour of day derived from step, ranging from 0 to 23.",
    "day_index": "Day number derived from the hourly step index.",
    "is_night": "Flag for transactions occurring during late-night hours.",
    "log_amount": "Log-transformed amount using log1p to reduce skew.",
    "amount_over_200k": "Flag for transactions above 200,000 units.",
    "destination_is_merchant": "Flag for destination accounts that look like merchants.",
    "destination_is_customer": "Flag for destination accounts that look like customers.",
    "origin_is_customer": "Flag for origin accounts that look like customers.",
    "is_transfer_or_cash_out": "Flag for transaction types most commonly associated with fraud in PaySim.",
}

EXCLUDED_FEATURE_DESCRIPTIONS = {
    "nameOrig": "Dropped from modeling because raw account IDs do not generalize well.",
    "nameDest": "Dropped from modeling because raw account IDs do not generalize well.",
    "oldbalanceOrg": "Excluded from modeling because the source dataset warns balance fields can leak label information.",
    "newbalanceOrig": "Excluded from modeling because the source dataset warns balance fields can leak label information.",
    "oldbalanceDest": "Excluded from modeling because the source dataset warns balance fields can leak label information.",
    "newbalanceDest": "Excluded from modeling because the source dataset warns balance fields can leak label information.",
}


def _default_description(feature_name: str) -> str:
    return "No description available."


def _resolve_base_feature(feature_name: str) -> str:
    known_features = (
        list(ENGINEERED_FEATURE_DESCRIPTIONS)
        + list(EXCLUDED_FEATURE_DESCRIPTIONS)
        + list(RAW_FEATURE_DESCRIPTIONS)
    )
    for candidate in sorted(known_features, key=len, reverse=True):
        if feature_name == candidate or feature_name.startswith(f"{candidate}_"):
            return candidate
    return feature_name


def describe_features(columns: list[str]) -> pd.DataFrame:
    """Build a readable feature dictionary for the notebook."""
    rows = []
    for column in columns:
        if column in ENGINEERED_FEATURE_DESCRIPTIONS:
            feature_group = "engineered"
            description = ENGINEERED_FEATURE_DESCRIPTIONS[column]
        elif column in EXCLUDED_FEATURE_DESCRIPTIONS:
            feature_group = "excluded"
            description = EXCLUDED_FEATURE_DESCRIPTIONS[column]
        elif column in RAW_FEATURE_DESCRIPTIONS:
            feature_group = "raw"
            description = RAW_FEATURE_DESCRIPTIONS[column]
        else:
            feature_group = "other"
            description = _default_description(column)
        rows.append(
            {
                "feature": column,
                "group": feature_group,
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def annotate_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Map transformed model feature names back to readable feature descriptions."""
    annotated = importance_df.copy()
    annotated["base_feature"] = (
        annotated["feature"]
        .str.replace(r"^(num|cat)__", "", regex=True)
        .str.replace(r"^encoder__", "", regex=True)
        .map(_resolve_base_feature)
    )
    descriptions = describe_features(annotated["base_feature"].tolist()).rename(
        columns={"feature": "base_feature"}
    )
    annotated = annotated.merge(descriptions, on="base_feature", how="left")
    return annotated[
        ["feature", "base_feature", "group", "description", "importance"]
    ]


def add_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create simple, interpretable features from the PaySim transaction fields."""
    enriched = frame.copy()

    if "step" in enriched.columns:
        enriched["hour_of_day"] = (enriched["step"] % 24).astype(int)
        enriched["day_index"] = (enriched["step"] // 24).astype(int) + 1
        enriched["is_night"] = enriched["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    if "amount" in enriched.columns:
        enriched["log_amount"] = np.log1p(enriched["amount"])
        enriched["amount_over_200k"] = enriched["amount"].ge(200_000).astype(int)

    if "nameDest" in enriched.columns:
        enriched["destination_is_merchant"] = enriched["nameDest"].str.startswith("M").fillna(False).astype(int)
        enriched["destination_is_customer"] = enriched["nameDest"].str.startswith("C").fillna(False).astype(int)

    if "nameOrig" in enriched.columns:
        enriched["origin_is_customer"] = enriched["nameOrig"].str.startswith("C").fillna(False).astype(int)

    if "type" in enriched.columns:
        enriched["is_transfer_or_cash_out"] = (
            enriched["type"].isin(["TRANSFER", "CASH_OUT"])
        ).astype(int)

    return enriched


def select_modeling_frame(frame: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Drop raw identifiers and leakage-prone fields before modeling."""
    drop_columns = [column for column in IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS if column in frame.columns]
    keep_columns = [column for column in frame.columns if column not in drop_columns]
    if target_col not in keep_columns:
        raise ValueError(f"Target column '{target_col}' was not found after column filtering.")
    return frame[keep_columns].copy()


def build_feature_lists(frame: pd.DataFrame, target_col: str = TARGET_COLUMN) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns for modeling."""
    features = frame.drop(columns=target_col)
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features
