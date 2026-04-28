"""
analysis.py

Final analysis script for IMDb sentiment classification capstone.

This script combines results from:
1. TF-IDF + Logistic Regression baseline
2. Lightweight CNN
3. DistilBERT fine-tuning

It produces:
- model comparison table
- review length slice analysis
- negation slice analysis
- representative error examples
- comparison figures
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score


# 0. Folder Setup

RESULTS_DIR = "Results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# 1. Helper Functions


def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Warning: file not found: {path}")
    return None


def find_bert_prediction_file():
    """
    Finds the DistilBERT test prediction CSV automatically.
    """
    pattern = os.path.join(RESULTS_DIR, "bert_predictions_*_test.csv")
    files = glob.glob(pattern)

    if len(files) == 0:
        print("Warning: no BERT test prediction file found.")
        return None

    # Use the first BERT test prediction file found.
    return files[0]


def add_text_features(df):
    """
    Adds review length and negation indicators.
    """
    df = df.copy()

    df["review_length"] = df["text"].astype(str).str.split().apply(len)

    df["has_negation"] = (
        df["text"]
        .astype(str)
        .str.lower()
        .str.contains(r"\bnot\b|\bnever\b|\bno\b|\bn't\b", regex=True)
    )

    return df


def compute_basic_metrics(df, model_name, dataset_note):
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    return {
        "model": model_name,
        "dataset_note": dataset_note,
        "num_examples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def compute_slice_metrics(df, model_name, dataset_note):
    """
    Computes accuracy and macro-F1 for:
    - short reviews
    - medium reviews
    - long reviews
    - negation reviews
    - non-negation reviews
    """

    df = add_text_features(df)

    q33 = df["review_length"].quantile(0.33)
    q66 = df["review_length"].quantile(0.66)

    slices = {
        "short_reviews": df[df["review_length"] <= q33],
        "medium_reviews": df[
            (df["review_length"] > q33) &
            (df["review_length"] <= q66)
        ],
        "long_reviews": df[df["review_length"] > q66],
        "negation_reviews": df[df["has_negation"] == True],
        "non_negation_reviews": df[df["has_negation"] == False],
    }

    rows = []

    for slice_name, subset in slices.items():
        if len(subset) == 0:
            continue

        rows.append({
            "model": model_name,
            "dataset_note": dataset_note,
            "slice": slice_name,
            "num_examples": len(subset),
            "accuracy": accuracy_score(
                subset["true_label"],
                subset["predicted_label"]
            ),
            "macro_f1": f1_score(
                subset["true_label"],
                subset["predicted_label"],
                average="macro"
            ),
            "avg_review_length": subset["review_length"].mean(),
        })

    return rows


def collect_error_examples(df, model_name, dataset_note, max_examples=10):
    """
    Collect representative wrong predictions.
    Prioritizes a mix of false positives and false negatives.
    """

    df = add_text_features(df)

    errors = df[df["true_label"] != df["predicted_label"]].copy()

    false_positive = errors[
        (errors["true_label"] == 0) &
        (errors["predicted_label"] == 1)
    ].head(max_examples // 2)

    false_negative = errors[
        (errors["true_label"] == 1) &
        (errors["predicted_label"] == 0)
    ].head(max_examples // 2)

    selected = pd.concat([false_positive, false_negative], axis=0)

    rows = []

    for _, row in selected.iterrows():
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            error_type = "false_positive"
        elif row["true_label"] == 1 and row["predicted_label"] == 0:
            error_type = "false_negative"
        else:
            error_type = "other"

        rows.append({
            "model": model_name,
            "dataset_note": dataset_note,
            "error_type": error_type,
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "review_length": row["review_length"],
            "has_negation": row["has_negation"],
            "text_excerpt": str(row["text"])[:800],
        })

    return rows


def save_bar_chart(df, metric, output_path, title):
    plt.figure(figsize=(9, 5))

    plt.bar(df["model"], df[metric])

    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to {output_path}")


def save_slice_chart(slice_df, slice_name, metric, output_path, title):
    subset = slice_df[slice_df["slice"] == slice_name].copy()

    if subset.empty:
        print(f"Warning: no data for slice {slice_name}")
        return

    plt.figure(figsize=(9, 5))

    plt.bar(subset["model"], subset[metric])

    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to {output_path}")


# 2. Load Prediction Files


print("Loading prediction files...")

prediction_sets = []

# Logistic Regression baseline
logistic_path = os.path.join(
    RESULTS_DIR,
    "baseline_predictions_logistic_regression_test.csv"
)

logistic_df = load_csv_if_exists(logistic_path)

if logistic_df is not None:
    prediction_sets.append({
        "model_name": "TF-IDF + Logistic Regression",
        "dataset_note": "Full IMDb test set",
        "df": logistic_df,
    })


# Naive Bayes baseline
naive_bayes_path = os.path.join(
    RESULTS_DIR,
    "baseline_predictions_naive_bayes_test.csv"
)

naive_bayes_df = load_csv_if_exists(naive_bayes_path)

if naive_bayes_df is not None:
    prediction_sets.append({
        "model_name": "TF-IDF + Naive Bayes",
        "dataset_note": "Full IMDb test set",
        "df": naive_bayes_df,
    })


# CNN
cnn_path = os.path.join(
    RESULTS_DIR,
    "cnn_predictions_test.csv"
)

cnn_df = load_csv_if_exists(cnn_path)

if cnn_df is not None:
    prediction_sets.append({
        "model_name": "Lightweight CNN",
        "dataset_note": "Full IMDb test set",
        "df": cnn_df,
    })


# DistilBERT
bert_path = find_bert_prediction_file()

if bert_path is not None:
    bert_df = load_csv_if_exists(bert_path)

    if bert_df is not None:
        prediction_sets.append({
            "model_name": "DistilBERT",
            "dataset_note": "5,000-example test subset",
            "df": bert_df,
        })


if len(prediction_sets) == 0:
    raise FileNotFoundError(
        "No prediction files were found. Run train_baseline.py, "
        "train_cnn.py, and train_bert.py first."
    )


# 3. Model Comparison Results

print("\nComputing model comparison results...")

comparison_rows = []

for item in prediction_sets:
    comparison_rows.append(
        compute_basic_metrics(
            df=item["df"],
            model_name=item["model_name"],
            dataset_note=item["dataset_note"],
        )
    )

comparison_df = pd.DataFrame(comparison_rows)

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_results.csv"
)

comparison_df.to_csv(comparison_path, index=False)

print("\nModel comparison results:")
print(comparison_df.to_string(index=False))
print(f"\nSaved model comparison results to {comparison_path}")

# 4. Slice Analysis

print("\nComputing slice analysis...")

slice_rows = []

for item in prediction_sets:
    slice_rows.extend(
        compute_slice_metrics(
            df=item["df"],
            model_name=item["model_name"],
            dataset_note=item["dataset_note"],
        )
    )

slice_df = pd.DataFrame(slice_rows)

slice_path = os.path.join(
    RESULTS_DIR,
    "slice_results.csv"
)

slice_df.to_csv(slice_path, index=False)

print("\nSlice analysis results:")
print(slice_df.to_string(index=False))
print(f"\nSaved slice results to {slice_path}")


# 5. Error Examples

print("\nCollecting error examples...")

error_rows = []

for item in prediction_sets:
    error_rows.extend(
        collect_error_examples(
            df=item["df"],
            model_name=item["model_name"],
            dataset_note=item["dataset_note"],
            max_examples=10,
        )
    )

error_df = pd.DataFrame(error_rows)

error_path = os.path.join(
    RESULTS_DIR,
    "error_examples.csv"
)

error_df.to_csv(error_path, index=False)

print(f"Saved representative error examples to {error_path}")


# 6. Figures

print("\nGenerating figures...")

accuracy_fig_path = os.path.join(
    FIGURES_DIR,
    "model_comparison_accuracy.png"
)

macro_f1_fig_path = os.path.join(
    FIGURES_DIR,
    "model_comparison_macro_f1.png"
)

save_bar_chart(
    df=comparison_df,
    metric="accuracy",
    output_path=accuracy_fig_path,
    title="IMDb Sentiment Classification Accuracy Comparison"
)

save_bar_chart(
    df=comparison_df,
    metric="macro_f1",
    output_path=macro_f1_fig_path,
    title="IMDb Sentiment Classification Macro-F1 Comparison"
)


length_fig_path = os.path.join(
    FIGURES_DIR,
    "length_slice_accuracy.png"
)

negation_fig_path = os.path.join(
    FIGURES_DIR,
    "negation_slice_accuracy.png"
)

save_slice_chart(
    slice_df=slice_df,
    slice_name="long_reviews",
    metric="accuracy",
    output_path=length_fig_path,
    title="Accuracy on Long Review Slice"
)

save_slice_chart(
    slice_df=slice_df,
    slice_name="negation_reviews",
    metric="accuracy",
    output_path=negation_fig_path,
    title="Accuracy on Negation Review Slice"
)


# 7. Summary for report

print("\n" + "=" * 60)
print("Report-ready summary")
print("=" * 60)

best_row = comparison_df.sort_values(by="macro_f1", ascending=False).iloc[0]

print(
    f"The best overall result in this analysis was achieved by "
    f"{best_row['model']}, with {best_row['accuracy']:.4f} accuracy "
    f"and {best_row['macro_f1']:.4f} macro-F1. "
)

print(
    "The TF-IDF and CNN models were evaluated on the full IMDb test set. "
    "The DistilBERT result was evaluated on a 5,000-example test subset due "
    "to CPU-only compute limitations, so it should be interpreted as a "
    "compute-limited transformer comparison rather than a fully controlled "
    "full-dataset comparison."
)

print(
    "Slice analysis results were saved for review-length groups and negation "
    "reviews. Representative false-positive and false-negative examples were "
    "also saved for qualitative error analysis."
)