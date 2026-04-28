"""
train_baseline.py

Baseline experiments for IMDb sentiment classification.

Models:
1. TF-IDF + Logistic Regression
2. TF-IDF + Naive Bayes

Outputs:
- results/baseline_results.csv
- results/baseline_predictions_logistic_regression_test.csv
- results/baseline_predictions_naive_bayes_test.csv
- figures/confusion_matrix_logistic_regression_test.png
- figures/confusion_matrix_naive_bayes_test.png
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


# ============================================================
# 0. Basic Setup
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# 1. Load Dataset
# ============================================================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Set this to True only for quick debugging.
USE_SUBSET = False

if USE_SUBSET:
    print("Using subset for quick test...")
    dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(2000))
    dataset["test"] = dataset["test"].shuffle(seed=SEED).select(range(1000))


# ============================================================
# 2. Train / Validation / Test Split
# ============================================================
# IMDb has official train and test sets.
# We split 10% of the training data as validation.

print("Creating train / validation / test split...")

split_dataset = dataset["train"].train_test_split(
    test_size=0.10,
    seed=SEED,
    stratify_by_column="label"
)

train_data = split_dataset["train"]
val_data = split_dataset["test"]
test_data = dataset["test"]

train_texts = train_data["text"]
train_labels = train_data["label"]

val_texts = val_data["text"]
val_labels = val_data["label"]

test_texts = test_data["text"]
test_labels = test_data["label"]

print(f"Train size: {len(train_texts)}")
print(f"Validation size: {len(val_texts)}")
print(f"Test size: {len(test_texts)}")


# ============================================================
# 3. TF-IDF Vectorization
# ============================================================

print("Building TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

print(f"TF-IDF feature shape: {X_train.shape}")


# ============================================================
# 4. Evaluation Function
# ============================================================

def evaluate_model(model_name, model, X, y, texts, split_name):
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    cm = confusion_matrix(y, preds)

    print("\n" + "=" * 60)
    print(f"{model_name} results on {split_name} set")
    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y,
            preds,
            target_names=["negative", "positive"],
            digits=4
        )
    )

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")

    # Save predictions
    pred_df = pd.DataFrame({
        "text": texts,
        "true_label": y,
        "predicted_label": preds,
        "correct": np.array(y) == np.array(preds)
    })

    pred_path = os.path.join(
        RESULTS_DIR,
        f"baseline_predictions_{safe_name}_{split_name}.csv"
    )
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} Confusion Matrix ({split_name})")
    plt.tight_layout()

    fig_path = os.path.join(
        FIGURES_DIR,
        f"confusion_matrix_{safe_name}_{split_name}.png"
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {fig_path}")

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "true_negative": cm[0, 0],
        "false_positive": cm[0, 1],
        "false_negative": cm[1, 0],
        "true_positive": cm[1, 1],
    }


# ============================================================
# 5. Model 1: Logistic Regression
# ============================================================

print("\nTraining Logistic Regression baseline...")

log_reg = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="liblinear",
    random_state=SEED
)

log_reg.fit(X_train, train_labels)

log_reg_val_result = evaluate_model(
    model_name="Logistic Regression",
    model=log_reg,
    X=X_val,
    y=val_labels,
    texts=val_texts,
    split_name="validation"
)

log_reg_test_result = evaluate_model(
    model_name="Logistic Regression",
    model=log_reg,
    X=X_test,
    y=test_labels,
    texts=test_texts,
    split_name="test"
)


# ============================================================
# 6. Model 2: Naive Bayes
# ============================================================

print("\nTraining Naive Bayes baseline...")

nb_model = MultinomialNB(alpha=1.0)

nb_model.fit(X_train, train_labels)

nb_val_result = evaluate_model(
    model_name="Naive Bayes",
    model=nb_model,
    X=X_val,
    y=val_labels,
    texts=val_texts,
    split_name="validation"
)

nb_test_result = evaluate_model(
    model_name="Naive Bayes",
    model=nb_model,
    X=X_test,
    y=test_labels,
    texts=test_texts,
    split_name="test"
)


# ============================================================
# 7. Save Summary Results
# ============================================================

results = [
    log_reg_val_result,
    log_reg_test_result,
    nb_val_result,
    nb_test_result
]

results_df = pd.DataFrame(results)

results_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("Final baseline results")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nSaved result summary to {results_path}")


# ============================================================
# 8. Report-Ready Summary
# ============================================================

test_results = results_df[results_df["split"] == "test"]
best_model = test_results.sort_values(by="macro_f1", ascending=False).iloc[0]

print("\nReport-ready summary:")
print(
    f"The best baseline model was {best_model['model']}, "
    f"which achieved {best_model['accuracy']:.4f} accuracy and "
    f"{best_model['macro_f1']:.4f} macro-F1 on the IMDb test set. "
    f"This result will be used as the baseline comparison point for "
    f"the CNN and BERT models."
)