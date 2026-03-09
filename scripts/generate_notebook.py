from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "fraud_detection_analysis.ipynb"


def md_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code_cell(code: str):
    return nbf.v4.new_code_cell(dedent(code).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    cells = [
        md_cell(
            """
            # Fraud Detection Analysis

            This notebook uses the **PaySim** fraud dataset, which has readable transaction fields such as:

            - `type`
            - `amount`
            - `nameOrig`
            - `nameDest`
            - `isFraud`
            - `isFlaggedFraud`
            """
        ),
        md_cell(
            """
            ## Problem framing

            Fraud detection is an **imbalanced classification** problem:

            - fraud is rare
            - missing fraud is costly
            - too many false positives create operational noise

            For that reason, this notebook focuses on:

            - precision-recall AUC
            - recall
            - precision
            - F-beta score with beta > 1
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

            PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            from src.data import TARGET_COLUMN, load_paysim_sample, train_valid_test_split
            from src.features import (
                LEAKAGE_COLUMNS,
                IDENTIFIER_COLUMNS,
                add_transaction_features,
                annotate_importance,
                build_feature_lists,
                describe_features,
                select_modeling_frame,
            )
            from src.modeling import (
                build_model_candidates,
                compare_models,
                compute_binary_metrics,
                extract_feature_importance,
                find_best_threshold,
            )

            pd.set_option("display.max_columns", 100)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.figsize"] = (12, 6)
            RANDOM_STATE = 42
            BETA = 2.0
            """
        ),
        md_cell(
            """
            ## 1. Load the dataset

            The PaySim CSV is large, so the loader creates a practical modeling sample:

            - keeps **all fraud rows**
            - keeps a reproducible sample of non-fraud rows
            - caches the sampled file in `data/processed`

            This keeps the notebook usable on a normal laptop while preserving the fraud cases.
            """
        ),
        code_cell(
            """
            paysim_df, sample_metadata = load_paysim_sample()

            pd.DataFrame([sample_metadata]).T.rename(columns={0: "value"})
            """
        ),
        code_cell(
            """
            print(f"Sample shape: {paysim_df.shape}")
            print(f"Sample fraud rate: {paysim_df[TARGET_COLUMN].mean():.4%}")

            paysim_df.head()
            """
        ),
        md_cell(
            """
            ## 2. Data dictionary and quality checks

            The dataset has meaningful field names, but not every field should be modeled directly:

            - `nameOrig` and `nameDest` are raw identifiers, not generalizable business features
            - the balance fields are excluded from modeling because the dataset documentation warns they can leak label information
            """
        ),
        code_cell(
            """
            describe_features(paysim_df.columns.tolist())
            """
        ),
        code_cell(
            """
            quality_checks = pd.DataFrame(
                {
                    "missing_values": paysim_df.isna().sum(),
                    "missing_pct": paysim_df.isna().mean() * 100,
                }
            ).sort_values("missing_values", ascending=False)

            duplicate_count = paysim_df.duplicated().sum()
            print(f"Duplicate rows in sample: {duplicate_count}")
            quality_checks
            """
        ),
        md_cell(
            """
            ## 3. Exploratory data analysis

            Start by understanding the class imbalance and the transaction type mix.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            class_counts = paysim_df[TARGET_COLUMN].value_counts().sort_index()
            sns.barplot(
                x=["Non-Fraud", "Fraud"],
                y=class_counts.values,
                palette=["#4C78A8", "#E45756"],
                ax=axes[0],
            )
            axes[0].set_title("Class Distribution in the Modeling Sample")
            axes[0].set_ylabel("Transactions")

            type_counts = paysim_df["type"].value_counts().reset_index()
            type_counts.columns = ["type", "transactions"]
            sns.barplot(data=type_counts, x="type", y="transactions", color="#4C78A8", ax=axes[1])
            axes[1].set_title("Transaction Type Distribution")
            axes[1].tick_params(axis="x", rotation=30)

            plt.tight_layout()
            """
        ),
        code_cell(
            """
            type_summary = (
                paysim_df.groupby("type")
                .agg(
                    transactions=(TARGET_COLUMN, "size"),
                    fraud_rate=(TARGET_COLUMN, "mean"),
                    avg_amount=("amount", "mean"),
                )
                .sort_values("fraud_rate", ascending=False)
                .reset_index()
            )

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            sns.barplot(data=type_summary, x="type", y="fraud_rate", color="#E45756", ax=axes[0])
            axes[0].set_title("Fraud Rate by Transaction Type")
            axes[0].tick_params(axis="x", rotation=30)

            sns.boxplot(
                data=paysim_df.assign(label=paysim_df[TARGET_COLUMN].map({0: "Non-Fraud", 1: "Fraud"})),
                x="label",
                y="amount",
                palette=["#4C78A8", "#E45756"],
                ax=axes[1],
            )
            axes[1].set_yscale("log")
            axes[1].set_title("Amount by Class (Log Scale)")

            plt.tight_layout()
            type_summary
            """
        ),
        code_cell(
            """
            eda_df = add_transaction_features(paysim_df)

            hourly_summary = (
                eda_df.groupby("hour_of_day")
                .agg(
                    transactions=(TARGET_COLUMN, "size"),
                    fraud_rate=(TARGET_COLUMN, "mean"),
                )
                .reset_index()
            )

            destination_summary = (
                eda_df.groupby("destination_is_merchant")[TARGET_COLUMN]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "fraud_rate", "count": "transactions"})
            )
            destination_summary["destination_group"] = destination_summary["destination_is_merchant"].map(
                {0: "Customer Destination", 1: "Merchant Destination"}
            )

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            sns.lineplot(data=hourly_summary, x="hour_of_day", y="fraud_rate", marker="o", color="#E45756", ax=axes[0])
            axes[0].set_title("Fraud Rate by Hour of Day")
            axes[0].set_xlabel("Hour of Day")

            sns.barplot(data=destination_summary, x="destination_group", y="fraud_rate", palette=["#4C78A8", "#54A24B"], ax=axes[1])
            axes[1].set_title("Fraud Rate by Destination Entity Type")
            axes[1].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            """
        ),
        md_cell(
            """
            ## 4. Feature engineering

            The engineered features stay close to business meaning:

            - hour and day features from `step`
            - log-transformed amount
            - large-amount flag
            - customer versus merchant destination
            - transfer or cash-out indicator
            """
        ),
        code_cell(
            """
            model_df = add_transaction_features(paysim_df)
            engineered_columns = sorted(set(model_df.columns) - set(paysim_df.columns))

            print("Engineered columns:", engineered_columns)
            model_df[engineered_columns + [TARGET_COLUMN]].describe().T
            """
        ),
        code_cell(
            """
            feature_dictionary = describe_features(model_df.columns.tolist())
            feature_dictionary
            """
        ),
        md_cell(
            """
            ## 5. Define the modeling dataset

            Two kinds of fields are intentionally removed before training:

            - raw account identifiers
            - balance columns flagged as leakage-prone by the dataset documentation
            """
        ),
        code_cell(
            """
            excluded_features = describe_features(IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS)
            excluded_features
            """
        ),
        code_cell(
            """
            modeling_df = select_modeling_frame(model_df)
            print(f"Modeling frame shape: {modeling_df.shape}")
            modeling_df.head()
            """
        ),
        md_cell(
            """
            ## 6. Train, validation, and test splits

            A stratified split keeps the fraud proportion consistent across subsets.
            """
        ),
        code_cell(
            """
            X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
                modeling_df,
                target_col=TARGET_COLUMN,
                test_size=0.2,
                valid_size=0.2,
                random_state=RANDOM_STATE,
            )

            split_summary = pd.DataFrame(
                {
                    "rows": [len(X_train), len(X_valid), len(X_test)],
                    "fraud_rate": [y_train.mean(), y_valid.mean(), y_test.mean()],
                },
                index=["train", "validation", "test"],
            )
            split_summary
            """
        ),
        md_cell(
            """
            ## 7. Model training

            Two models are compared:

            - logistic regression as a strong linear baseline
            - random forest as a nonlinear model with feature importance support
            """
        ),
        code_cell(
            """
            numeric_features, categorical_features = build_feature_lists(modeling_df, target_col=TARGET_COLUMN)
            models = build_model_candidates(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_state=RANDOM_STATE,
            )

            comparison_df, fitted_models = compare_models(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                beta=BETA,
            )

            comparison_df
            """
        ),
        md_cell(
            """
            ## 8. Threshold tuning

            The default 0.50 threshold is rarely optimal for fraud problems.
            Threshold tuning helps control the tradeoff between catching more fraud and generating false alerts.
            """
        ),
        code_cell(
            """
            best_model_name = comparison_df.iloc[0]["model"]
            best_model = fitted_models[best_model_name].pipeline
            valid_scores = fitted_models[best_model_name].validation_scores

            best_threshold, threshold_frame = find_best_threshold(
                y_true=y_valid,
                y_scores=valid_scores,
                beta=BETA,
                min_precision=0.10,
            )

            print(f"Selected model: {best_model_name}")
            print(f"Best validation threshold: {best_threshold:.2f}")

            threshold_frame.sort_values("f_beta", ascending=False).head(10)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            sns.lineplot(data=threshold_frame, x="threshold", y="precision", ax=axes[0], label="Precision")
            sns.lineplot(data=threshold_frame, x="threshold", y="recall", ax=axes[0], label="Recall")
            axes[0].axvline(best_threshold, color="black", linestyle="--")
            axes[0].set_title("Precision and Recall vs Threshold")

            sns.lineplot(data=threshold_frame, x="threshold", y="f_beta", ax=axes[1], color="#E45756")
            axes[1].axvline(best_threshold, color="black", linestyle="--")
            axes[1].set_title("F-beta vs Threshold")

            valid_metrics_default = compute_binary_metrics(y_valid, valid_scores, threshold=0.50, beta=BETA)
            valid_metrics_tuned = compute_binary_metrics(y_valid, valid_scores, threshold=best_threshold, beta=BETA)
            comparison_plot = pd.DataFrame(
                [
                    {"setting": "0.50 threshold", **valid_metrics_default},
                    {"setting": f"{best_threshold:.2f} threshold", **valid_metrics_tuned},
                ]
            )
            comparison_melted = comparison_plot.melt(
                id_vars="setting",
                value_vars=["precision", "recall", "f_beta"],
                var_name="metric",
                value_name="score",
            )
            sns.barplot(data=comparison_melted, x="metric", y="score", hue="setting", ax=axes[2])
            axes[2].set_title("Validation Metrics Before vs After Tuning")

            plt.tight_layout()
            """
        ),
        md_cell(
            """
            ## 9. Final test set evaluation

            The test set stays untouched until model selection and threshold tuning are complete.
            """
        ),
        code_cell(
            """
            test_scores = best_model.predict_proba(X_test)[:, 1]
            test_metrics = compute_binary_metrics(
                y_true=y_test,
                y_scores=test_scores,
                threshold=best_threshold,
                beta=BETA,
            )

            pd.DataFrame([test_metrics]).T.rename(columns={0: "test_value"})
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            RocCurveDisplay.from_predictions(
                y_test,
                test_scores,
                ax=axes[0],
                curve_kwargs={"color": "#4C78A8"},
            )
            axes[0].set_title("ROC Curve")

            PrecisionRecallDisplay.from_predictions(
                y_test,
                test_scores,
                ax=axes[1],
                curve_kwargs={"color": "#E45756"},
            )
            axes[1].set_title("Precision-Recall Curve")

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                (test_scores >= best_threshold).astype(int),
                ax=axes[2],
                cmap="Blues",
                colorbar=False,
                display_labels=["Non-Fraud", "Fraud"],
            )
            axes[2].set_title(f"Confusion Matrix at Threshold = {best_threshold:.2f}")
            axes[2].set_xlabel("Predicted class")
            axes[2].set_ylabel("Actual class")

            plt.tight_layout()
            """
        ),
        code_cell(
            """
            tn = int(test_metrics["true_negatives"])
            fp = int(test_metrics["false_positives"])
            fn = int(test_metrics["false_negatives"])
            tp = int(test_metrics["true_positives"])

            confusion_explanation = pd.DataFrame(
                [
                    {"cell": "Top-left", "meaning": "Correctly predicted non-fraud", "count": tn},
                    {"cell": "Top-right", "meaning": "Non-fraud predicted as fraud", "count": fp},
                    {"cell": "Bottom-left", "meaning": "Fraud predicted as non-fraud", "count": fn},
                    {"cell": "Bottom-right", "meaning": "Correctly predicted fraud", "count": tp},
                ]
            )

            confusion_explanation
            """
        ),
        code_cell(
            """
            confusion_rate_summary = pd.DataFrame(
                [
                    {"metric": "Recall (fraud caught)", "value": tp / (tp + fn) if (tp + fn) else 0.0},
                    {"metric": "Precision (alerts that were truly fraud)", "value": tp / (tp + fp) if (tp + fp) else 0.0},
                    {"metric": "False positive rate on normal transactions", "value": fp / (fp + tn) if (fp + tn) else 0.0},
                    {"metric": "Miss rate on fraud transactions", "value": fn / (fn + tp) if (fn + tp) else 0.0},
                ]
            )

            confusion_rate_summary
            """
        ),
        md_cell(
            """
            How to read the confusion matrix:

            - the rows are the **actual** class
            - the columns are the **predicted** class
            - the top-left cell is good because those are normal transactions correctly left alone
            - the bottom-right cell is good because those are fraud cases correctly caught
            - the top-right cell is the false-alarm count
            - the bottom-left cell is the missed-fraud count

            In this problem, the bottom-left cell is especially important because those are fraud cases the model failed to catch.
            """
        ),
        md_cell(
            """
            ## 10. Model interpretation

            The importance table below maps transformed model features back to readable source features.
            That means a value such as `type_TRANSFER` is grouped under the base feature `type`,
            rather than appearing as an unexplained model artifact.
            """
        ),
        code_cell(
            """
            feature_names = best_model.named_steps["preprocessor"].get_feature_names_out().tolist()
            importance_df = extract_feature_importance(best_model, feature_names)
            annotated_importance_df = annotate_importance(importance_df)
            annotated_importance_df.head(15)
            """
        ),
        code_cell(
            """
            top_importance = annotated_importance_df.head(15).sort_values("importance", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(top_importance["base_feature"], top_importance["importance"], color="#54A24B")
            ax.set_title(f"Top 15 Features for {best_model_name}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            """
        ),
        md_cell(
            """
            Interpretation notes:

            - `type` tells us which transaction category is most predictive
            - `amount` and `log_amount` capture size effects
            - `hour_of_day` and `is_night` capture timing behavior
            - destination and transfer indicators help explain where risky flows concentrate

            On this chart, a larger bar means the model relied more on that feature when splitting transactions into fraud and non-fraud groups.
            The importance value does **not** tell direction by itself. For example, a large importance for `type` means transaction category matters a lot,
            not that every value of `type` increases fraud risk.
            """
        ),
        md_cell(
            """
            ## 11. Export predictions

            The scored test transactions can be exported for review or downstream analysis.
            """
        ),
        code_cell(
            """
            prediction_frame = X_test.copy()
            prediction_frame["actual_class"] = y_test.values
            prediction_frame["fraud_probability"] = test_scores
            prediction_frame["predicted_class"] = (test_scores >= best_threshold).astype(int)
            prediction_frame = prediction_frame.sort_values("fraud_probability", ascending=False)

            output_path = PROJECT_ROOT / "data" / "processed" / "test_set_predictions.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_frame.to_csv(output_path, index=False)

            print(f"Saved predictions to: {output_path}")
            prediction_frame.head(10)
            """
        ),
        md_cell(
            """
            ## 12. Final takeaways

            Key lessons from this analysis:

            - readable feature names make the model easier to interpret
            - excluding leakage-prone columns matters more than squeezing out a little extra score
            - threshold tuning is part of the modeling problem, not an afterthought
            - exporting scored transactions makes the workflow easier to inspect and extend
            """
        ),
    ]

    nb["cells"] = cells
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as notebook_file:
        nbf.write(notebook, notebook_file)


if __name__ == "__main__":
    main()
