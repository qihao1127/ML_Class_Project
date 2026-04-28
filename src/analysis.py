import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0. Settings
# =========================

RESULTS_DIR = "Results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================
# 1. Helper Functions
# =========================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def compute_accuracy(df):
    return (df["true_label"] == df["predicted_label"]).mean()


def count_words(text):
    return len(str(text).split())


def contains_negation(text):
    text = str(text).lower()

    negation_patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bnothing\b",
        r"\bnowhere\b",
        r"\bneither\b",
        r"\bnor\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdon't\b",
        r"\bdidn't\b",
        r"\bisn't\b",
        r"\bwasn't\b",
        r"\baren't\b",
        r"\bweren't\b",
        r"\bwon't\b",
        r"\bwouldn't\b",
        r"\bshouldn't\b",
        r"\bcouldn't\b",
        r"n't\b",
    ]

    return any(re.search(pattern, text) for pattern in negation_patterns)


def plot_bar(df, x_col, y_col, title, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure to {output_path}")


# =========================
# 2. Model Comparison
# =========================

print("Loading summary result files...")

baseline_results = load_csv(
    os.path.join(RESULTS_DIR, "baseline_results_subset_2000.csv")
)

cnn_results = load_csv(
    os.path.join(RESULTS_DIR, "cnn_results_subset_2000.csv")
)

bert_results = load_csv(
    os.path.join(RESULTS_DIR, "bert_results_subset_2000.csv")
)

all_results = pd.concat(
    [
        baseline_results,
        cnn_results,
        bert_results,
    ],
    ignore_index=True,
)

test_results = all_results[all_results["split"] == "test"].copy()

model_name_map = {
    "Logistic Regression": "Logistic Regression",
    "Naive Bayes": "Naive Bayes",
    "CNN": "Lightweight CNN",
    "DistilBERT": "DistilBERT",
}

test_results["model_display"] = test_results["model"].map(model_name_map)

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_results_subset_2000.csv",
)

test_results.to_csv(comparison_path, index=False)

print("\nModel comparison results:")
print(test_results)
print(f"\nSaved model comparison results to {comparison_path}")


# Accuracy plot
plot_bar(
    df=test_results,
    x_col="model_display",
    y_col="accuracy",
    title="Model Comparison on IMDb Test Set - Subset 2000",
    ylabel="Test Accuracy",
    output_path=os.path.join(
        FIGURES_DIR,
        "model_comparison_accuracy_subset_2000.png",
    ),
)

# Macro-F1 plot
plot_bar(
    df=test_results,
    x_col="model_display",
    y_col="macro_f1",
    title="Model Comparison on IMDb Test Set - Subset 2000",
    ylabel="Test Macro-F1",
    output_path=os.path.join(
        FIGURES_DIR,
        "model_comparison_macro_f1_subset_2000.png",
    ),
)


# =========================
# 3. Load Prediction Files
# =========================

print("\nLoading prediction files...")

prediction_files = {
    "Logistic Regression": os.path.join(
        RESULTS_DIR,
        "baseline_predictions_logistic_regression_subset_2000_test.csv",
    ),
    "Naive Bayes": os.path.join(
        RESULTS_DIR,
        "baseline_predictions_naive_bayes_subset_2000_test.csv",
    ),
    "Lightweight CNN": os.path.join(
        RESULTS_DIR,
        "cnn_predictions_subset_2000_test.csv",
    ),
    "DistilBERT": os.path.join(
        RESULTS_DIR,
        "bert_predictions_distilbert_fine_tuned_subset_2000_test.csv",
    ),
}

prediction_dfs = {}

for model_name, path in prediction_files.items():
    df = load_csv(path)
    prediction_dfs[model_name] = df
    print(f"{model_name}: {len(df)} test examples")


# =========================
# 4. Length Slice Analysis
# =========================

print("\nRunning length slice analysis...")

length_rows = []

for model_name, df in prediction_dfs.items():
    df = df.copy()
    df["review_length"] = df["text"].apply(count_words)

    df["length_group"] = pd.cut(
        df["review_length"],
        bins=[0, 100, 250, 500, np.inf],
        labels=[
            "short_0_100",
            "medium_101_250",
            "long_251_500",
            "very_long_500_plus",
        ],
        include_lowest=True,
    )

    for group_name, group_df in df.groupby("length_group", observed=False):
        if len(group_df) == 0:
            continue

        length_rows.append(
            {
                "model": model_name,
                "slice_type": "review_length",
                "slice_name": str(group_name),
                "num_examples": len(group_df),
                "accuracy": compute_accuracy(group_df),
            }
        )

length_slice_df = pd.DataFrame(length_rows)

length_slice_path = os.path.join(
    RESULTS_DIR,
    "length_slice_results_subset_2000.csv",
)

length_slice_df.to_csv(length_slice_path, index=False)

print("\nLength slice results:")
print(length_slice_df)
print(f"\nSaved length slice results to {length_slice_path}")


# Plot length slice
plt.figure(figsize=(10, 6))

for model_name in length_slice_df["model"].unique():
    model_df = length_slice_df[length_slice_df["model"] == model_name]
    plt.plot(
        model_df["slice_name"],
        model_df["accuracy"],
        marker="o",
        label=model_name,
    )

plt.title("Accuracy by Review Length - Subset 2000")
plt.xlabel("Review Length Group")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.xticks(rotation=25, ha="right")
plt.legend()
plt.tight_layout()

length_plot_path = os.path.join(
    FIGURES_DIR,
    "length_slice_accuracy_subset_2000.png",
)

plt.savefig(length_plot_path, dpi=300)
plt.close()

print(f"Saved figure to {length_plot_path}")


# =========================
# 5. Negation Slice Analysis
# =========================

print("\nRunning negation slice analysis...")

negation_rows = []

for model_name, df in prediction_dfs.items():
    df = df.copy()
    df["contains_negation"] = df["text"].apply(contains_negation)

    for slice_value, group_df in df.groupby("contains_negation"):
        slice_name = (
            "contains_negation"
            if slice_value
            else "no_negation"
        )

        negation_rows.append(
            {
                "model": model_name,
                "slice_type": "negation",
                "slice_name": slice_name,
                "num_examples": len(group_df),
                "accuracy": compute_accuracy(group_df),
            }
        )

negation_slice_df = pd.DataFrame(negation_rows)

negation_slice_path = os.path.join(
    RESULTS_DIR,
    "negation_slice_results_subset_2000.csv",
)

negation_slice_df.to_csv(negation_slice_path, index=False)

print("\nNegation slice results:")
print(negation_slice_df)
print(f"\nSaved negation slice results to {negation_slice_path}")


# Plot negation slice
plt.figure(figsize=(8, 5))

for model_name in negation_slice_df["model"].unique():
    model_df = negation_slice_df[
        negation_slice_df["model"] == model_name
    ]

    plt.plot(
        model_df["slice_name"],
        model_df["accuracy"],
        marker="o",
        label=model_name,
    )

plt.title("Accuracy on Negation Slice - Subset 2000")
plt.xlabel("Slice")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()

negation_plot_path = os.path.join(
    FIGURES_DIR,
    "negation_slice_accuracy_subset_2000.png",
)

plt.savefig(negation_plot_path, dpi=300)
plt.close()

print(f"Saved figure to {negation_plot_path}")


# =========================
# 6. Combined Slice Results
# =========================

slice_results = pd.concat(
    [
        length_slice_df,
        negation_slice_df,
    ],
    ignore_index=True,
)

slice_results_path = os.path.join(
    RESULTS_DIR,
    "slice_results_subset_2000.csv",
)

slice_results.to_csv(slice_results_path, index=False)

print(f"\nSaved combined slice results to {slice_results_path}")


# =========================
# 7. Error Examples
# =========================

print("\nSaving error examples...")

error_rows = []

for model_name, df in prediction_dfs.items():
    df = df.copy()
    df["correct"] = df["true_label"] == df["predicted_label"]

    errors = df[df["correct"] == False].head(10)

    for _, row in errors.iterrows():
        error_rows.append(
            {
                "model": model_name,
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "text": row["text"],
            }
        )

error_examples_df = pd.DataFrame(error_rows)

error_examples_path = os.path.join(
    RESULTS_DIR,
    "error_examples_subset_2000.csv",
)

error_examples_df.to_csv(error_examples_path, index=False)

print(f"Saved error examples to {error_examples_path}")


# =========================
# 8. Done
# =========================

print("\nAnalysis complete.")