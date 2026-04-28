import os
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import (
    set_seed,
    make_dirs,
    evaluate_predictions,
    save_predictions,
    save_confusion_matrix,
    safe_name,
)


# =========================
# 0. Settings
# =========================

SEED = 42

RESULTS_DIR = "Results"
FIGURES_DIR = "figures"
MODELS_DIR = "models"

USE_SUBSET = True
SUBSET_TRAIN_SIZE = 2000
SUBSET_VAL_SIZE = 500
SUBSET_TEST_SIZE = 1000

set_seed(SEED)
make_dirs(RESULTS_DIR, FIGURES_DIR, MODELS_DIR)


# =========================
# 1. Load IMDb Dataset
# =========================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Shuffle first to avoid class-order issues
train_data = dataset["train"].shuffle(seed=SEED)
test_data = dataset["test"].shuffle(seed=SEED)

if USE_SUBSET:
    train_subset = train_data.select(
        range(SUBSET_TRAIN_SIZE + SUBSET_VAL_SIZE)
    )
    test_subset = test_data.select(range(SUBSET_TEST_SIZE))

    train_texts = [
        str(x) for x in train_subset["text"][:SUBSET_TRAIN_SIZE]
    ]
    train_labels = list(train_subset["label"][:SUBSET_TRAIN_SIZE])

    val_texts = [
        str(x)
        for x in train_subset["text"][
            SUBSET_TRAIN_SIZE : SUBSET_TRAIN_SIZE + SUBSET_VAL_SIZE
        ]
    ]
    val_labels = list(
        train_subset["label"][
            SUBSET_TRAIN_SIZE : SUBSET_TRAIN_SIZE + SUBSET_VAL_SIZE
        ]
    )

    test_texts = [str(x) for x in test_subset["text"]]
    test_labels = list(test_subset["label"])

else:
    VALIDATION_SIZE = 5000

    train_texts_all = [str(x) for x in train_data["text"]]
    train_labels_all = list(train_data["label"])

    train_texts = train_texts_all[:-VALIDATION_SIZE]
    train_labels = train_labels_all[:-VALIDATION_SIZE]

    val_texts = train_texts_all[-VALIDATION_SIZE:]
    val_labels = train_labels_all[-VALIDATION_SIZE:]

    test_texts = [str(x) for x in test_data["text"]]
    test_labels = list(test_data["label"])

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

print("Train label counts:", np.bincount(train_labels))
print("Validation label counts:", np.bincount(val_labels))
print("Test label counts:", np.bincount(test_labels))


# =========================
# 2. Define Baseline Models
# =========================

models = {
    "Logistic Regression": Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "Naive Bayes": Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    ),
}


# =========================
# 3. Train and Evaluate
# =========================

all_results = []

for model_name, model in models.items():
    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    model.fit(train_texts, train_labels)

    model_id = safe_name(model_name)

    for split_name, texts, labels in [
        ("validation", val_texts, val_labels),
        ("test", test_texts, test_labels),
    ]:
        preds = model.predict(texts)

        result = evaluate_predictions(
            y_true=labels,
            y_pred=preds,
            model_name=model_name,
            split_name=split_name,
        )

        all_results.append(
            {
                "model": model_name,
                "split": split_name,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "true_negative": result["true_negative"],
                "false_positive": result["false_positive"],
                "false_negative": result["false_negative"],
                "true_positive": result["true_positive"],
            }
        )

        save_predictions(
            texts=texts,
            labels=labels,
            preds=preds,
            output_path=os.path.join(
                RESULTS_DIR,
                f"baseline_predictions_{model_id}_subset_2000_{split_name}.csv",
            ),
        )

        save_confusion_matrix(
            cm=result["confusion_matrix"],
            model_name=model_name,
            split_name=split_name,
            output_path=os.path.join(
                FIGURES_DIR,
                f"baseline_confusion_matrix_{model_id}_subset_2000_{split_name}.png",
            ),
        )


# =========================
# 4. Save Summary Results
# =========================

results_df = pd.DataFrame(all_results)

if USE_SUBSET:
    results_path = os.path.join(
        RESULTS_DIR,
        "baseline_results_subset_2000.csv",
    )
else:
    results_path = os.path.join(
        RESULTS_DIR,
        "baseline_results.csv",
    )

results_df.to_csv(results_path, index=False)

print("\nBaseline results:")
print(results_df)
print(f"\nSaved baseline results to {results_path}")