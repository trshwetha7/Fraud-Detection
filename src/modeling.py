from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    validation_scores: np.ndarray
    metrics: dict[str, float]


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a shared preprocessing pipeline for all models."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_model_candidates(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Create a small but strong candidate set for imbalanced fraud detection."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_samples=0.5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        for name, estimator in models.items()
    }


def compute_binary_metrics(
    y_true: pd.Series,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    beta: float = 2.0,
) -> dict[str, float]:
    """Calculate metrics that matter for fraud detection."""
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc": average_precision_score(y_true, y_scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f_beta": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_negatives": float(tn),
    }


def compare_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    beta: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, ModelResult]]:
    """Train each model and compare them on the validation set."""
    rows: list[dict[str, float | str]] = []
    fitted_results: dict[str, ModelResult] = {}

    for name, pipeline in models.items():
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)
        scores = fitted.predict_proba(X_valid)[:, 1]
        metrics = compute_binary_metrics(y_valid, scores, threshold=0.5, beta=beta)
        rows.append({"model": name, **metrics})
        fitted_results[name] = ModelResult(
            name=name,
            pipeline=fitted,
            validation_scores=scores,
            metrics=metrics,
        )

    comparison = pd.DataFrame(rows).sort_values(
        by=["pr_auc", "recall"],
        ascending=False,
    )
    return comparison, fitted_results


def find_best_threshold(
    y_true: pd.Series,
    y_scores: np.ndarray,
    beta: float = 2.0,
    min_precision: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """
    Search thresholds and keep the one with the highest F-beta score.

    `min_precision` prevents extremely low-precision thresholds from being chosen.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_scores, threshold=threshold, beta=beta)
        rows.append({"threshold": threshold, **metrics})

    threshold_frame = pd.DataFrame(rows)
    eligible = threshold_frame[threshold_frame["precision"] >= min_precision]
    if eligible.empty:
        eligible = threshold_frame

    best_row = eligible.sort_values(
        by=["f_beta", "recall", "precision"],
        ascending=False,
    ).iloc[0]
    return float(best_row["threshold"]), threshold_frame


def extract_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Return model-specific importance values for interpretation."""
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise ValueError("The supplied model does not expose feature importance.")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
