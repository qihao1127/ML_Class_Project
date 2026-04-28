"""
train_bert.py

CPU-friendly DistilBERT fine-tuning experiment for IMDb sentiment classification.

This script uses a subset of IMDb because full BERT training on CPU is very slow.

Model:
- DistilBERT fine-tuned on IMDb sentiment classification

Outputs:
- results/bert_results.csv
- results/bert_predictions_distilbert_fine_tuned_lr_2e_5_epochs_2_validation.csv
- results/bert_predictions_distilbert_fine_tuned_lr_2e_5_epochs_2_test.csv
- figures/confusion_matrix_bert_distilbert_fine_tuned_lr_2e_5_epochs_2_validation.png
- figures/confusion_matrix_bert_distilbert_fine_tuned_lr_2e_5_epochs_2_test.png
- models/distilbert_fine_tuned_lr_2e_5_epochs_2/
"""

import os
import random
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

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
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU. DistilBERT + subset mode is used for feasibility.")


# ============================================================
# 1. Experiment Settings
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16

USE_SUBSET = True
SUBSET_TRAIN_SIZE = 5000
SUBSET_TEST_SIZE = 5000

EXPERIMENT_NAME = "DistilBERT Fine Tuned LR 2e-5 Epochs 2"
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2


# ============================================================
# 2. Helper Functions
# ============================================================

def safe_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("=", "")
        .replace(".", "")
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def build_training_args(output_dir):
    """
    Handles different transformers versions.
    Some versions use eval_strategy, older versions use evaluation_strategy.
    """

    params = inspect.signature(TrainingArguments.__init__).parameters

    args_dict = {
        "output_dir": output_dir,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "report_to": "none",
        "seed": SEED,
        "fp16": torch.cuda.is_available(),
        "save_total_limit": 1,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
    }

    if "eval_strategy" in params:
        args_dict["eval_strategy"] = "epoch"
    else:
        args_dict["evaluation_strategy"] = "epoch"

    return TrainingArguments(**args_dict)


def save_predictions(texts, labels, preds, experiment_name, split_name):
    pred_df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "predicted_label": preds,
        "correct": np.array(labels) == np.array(preds),
    })

    file_name = f"bert_predictions_{safe_name(experiment_name)}_{split_name}.csv"
    pred_path = os.path.join(RESULTS_DIR, file_name)

    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")


def save_confusion_matrix(cm, experiment_name, split_name):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{experiment_name} Confusion Matrix ({split_name})")
    plt.tight_layout()

    file_name = f"confusion_matrix_bert_{safe_name(experiment_name)}_{split_name}.png"
    fig_path = os.path.join(FIGURES_DIR, file_name)

    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {fig_path}")


def evaluate_and_save(trainer, tokenized_dataset, raw_texts, labels, experiment_name, split_name):
    print("\n" + "=" * 60)
    print(f"Evaluating {experiment_name} on {split_name} set")
    print("=" * 60)

    prediction_output = trainer.predict(tokenized_dataset)
    logits = prediction_output.predictions
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=["negative", "positive"],
            digits=4,
        )
    )

    save_predictions(
        texts=raw_texts,
        labels=labels,
        preds=preds,
        experiment_name=experiment_name,
        split_name=split_name,
    )

    save_confusion_matrix(
        cm=cm,
        experiment_name=experiment_name,
        split_name=split_name,
    )

    return {
        "model": experiment_name,
        "base_model": MODEL_NAME,
        "split": split_name,
        "train_subset_size": SUBSET_TRAIN_SIZE if USE_SUBSET else "full",
        "test_subset_size": SUBSET_TEST_SIZE if USE_SUBSET else "full",
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "true_negative": cm[0, 0],
        "false_positive": cm[0, 1],
        "false_negative": cm[1, 0],
        "true_positive": cm[1, 1],
    }


# ============================================================
# 3. Load IMDb Dataset
# ============================================================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

if USE_SUBSET:
    print(f"Using subset: train={SUBSET_TRAIN_SIZE}, test={SUBSET_TEST_SIZE}")
    dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(SUBSET_TRAIN_SIZE))
    dataset["test"] = dataset["test"].shuffle(seed=SEED).select(range(SUBSET_TEST_SIZE))


# ============================================================
# 4. Train / Validation / Test Split
# ============================================================

print("Creating train / validation / test split...")

split_dataset = dataset["train"].train_test_split(
    test_size=0.10,
    seed=SEED,
    stratify_by_column="label",
)

train_data = split_dataset["train"]
val_data = split_dataset["test"]
test_data = dataset["test"]

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")


# ============================================================
# 5. Tokenization
# ============================================================

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("Tokenizing datasets...")

tokenized_train = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

tokenized_val = val_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

tokenized_test = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_val = tokenized_val.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

tokenized_train.set_format("torch")
tokenized_val.set_format("torch")
tokenized_test.set_format("torch")


# ============================================================
# 6. Model and Trainer
# ============================================================

print("\n" + "#" * 80)
print(f"Running experiment: {EXPERIMENT_NAME}")
print("#" * 80)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

output_dir = os.path.join(MODELS_DIR, safe_name(EXPERIMENT_NAME))

training_args = build_training_args(output_dir=output_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)


# ============================================================
# 7. Fine-Tuning
# ============================================================

print("Fine-tuning DistilBERT...")
trainer.train()

print(f"Saving final model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)


# ============================================================
# 8. Evaluation
# ============================================================

val_result = evaluate_and_save(
    trainer=trainer,
    tokenized_dataset=tokenized_val,
    raw_texts=val_data["text"],
    labels=val_data["label"],
    experiment_name=EXPERIMENT_NAME,
    split_name="validation",
)

test_result = evaluate_and_save(
    trainer=trainer,
    tokenized_dataset=tokenized_test,
    raw_texts=test_data["text"],
    labels=test_data["label"],
    experiment_name=EXPERIMENT_NAME,
    split_name="test",
)


# ============================================================
# 9. Save Summary Results
# ============================================================

results_df = pd.DataFrame([val_result, test_result])

results_path = os.path.join(RESULTS_DIR, "bert_results.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("Final DistilBERT results")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nSaved BERT result summary to {results_path}")


# ============================================================
# 10. Report-Ready Summary
# ============================================================

test_row = results_df[results_df["split"] == "test"].iloc[0]

print("\nReport-ready summary:")
print(
    f"The DistilBERT fine-tuning experiment used a {SUBSET_TRAIN_SIZE}-example "
    f"training subset and a {SUBSET_TEST_SIZE}-example test subset due to CPU-only "
    f"compute limitations. The model achieved {test_row['accuracy']:.4f} accuracy "
    f"and {test_row['macro_f1']:.4f} macro-F1 on the IMDb test subset. "
    f"This result will be compared against the full-dataset TF-IDF Logistic Regression "
    f"baseline and lightweight CNN model, with the subset limitation clearly stated."
)