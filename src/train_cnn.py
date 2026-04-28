"""
train_cnn.py

Lightweight CNN experiment for IMDb sentiment classification.

Model:
Embedding -> 1D Convolution -> ReLU -> Max Pooling -> Dropout -> Linear Classifier

Outputs:
- results/cnn_results.csv
- results/cnn_predictions_validation.csv
- results/cnn_predictions_test.csv
- figures/confusion_matrix_cnn_validation.png
- figures/confusion_matrix_cnn_test.png
- models/cnn_model.pt
"""

import os
import re
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

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


# ============================================================
# 1. Hyperparameters
# ============================================================

USE_SUBSET = False

MAX_VOCAB_SIZE = 30000
MAX_LENGTH = 256

EMBED_DIM = 100
NUM_FILTERS = 128
KERNEL_SIZE = 3
DROPOUT = 0.5

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


# ============================================================
# 2. Text Processing
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return clean_text(text).split()


def build_vocab(texts, max_vocab_size):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids = ids + [vocab["<PAD>"]] * (max_length - len(ids))

    return ids


# ============================================================
# 3. Dataset Class
# ============================================================

class IMDbCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        input_ids = encode_text(
            self.texts[index],
            self.vocab,
            self.max_length
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


# ============================================================
# 4. CNN Model
# ============================================================

class LightweightCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_filters,
        kernel_size,
        dropout,
        num_classes=2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        # CNN expects shape: batch, channels, sequence length
        x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        x = self.relu(x)

        # Global max pooling over sequence length
        x = torch.max(x, dim=2)[0]

        x = self.dropout(x)
        logits = self.fc(x)

        return logits


# ============================================================
# 5. Evaluation Function
# ============================================================

def evaluate_model(model, dataloader, split_name):
    model.eval()

    all_labels = []
    all_preds = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * input_ids.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print(f"CNN results on {split_name} set")
    print("=" * 60)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["negative", "positive"],
            digits=4,
        )
    )

    return {
        "split": split_name,
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "true_negative": cm[0, 0],
        "false_positive": cm[0, 1],
        "false_negative": cm[1, 0],
        "true_positive": cm[1, 1],
        "labels": all_labels,
        "preds": all_preds,
        "confusion_matrix": cm,
    }


def save_predictions(texts, labels, preds, split_name):
    pred_df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "predicted_label": preds,
        "correct": np.array(labels) == np.array(preds),
    })

    pred_path = os.path.join(
        RESULTS_DIR,
        f"cnn_predictions_{split_name}.csv"
    )

    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")


def save_confusion_matrix(cm, split_name):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"CNN Confusion Matrix ({split_name})")
    plt.tight_layout()

    fig_path = os.path.join(
        FIGURES_DIR,
        f"confusion_matrix_cnn_{split_name}.png"
    )

    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {fig_path}")


# ============================================================
# 6. Load IMDb Dataset
# ============================================================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

if USE_SUBSET:
    print("Using subset for quick testing...")
    dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(3000))
    dataset["test"] = dataset["test"].shuffle(seed=SEED).select(range(1000))


# ============================================================
# 7. Train / Validation / Test Split
# ============================================================

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
# 8. Build Vocabulary
# ============================================================

print("Building vocabulary from training data...")

vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")


# ============================================================
# 9. Build Datasets and Dataloaders
# ============================================================

train_dataset = IMDbCNNDataset(
    train_texts,
    train_labels,
    vocab,
    MAX_LENGTH
)

val_dataset = IMDbCNNDataset(
    val_texts,
    val_labels,
    vocab,
    MAX_LENGTH
)

test_dataset = IMDbCNNDataset(
    test_texts,
    test_labels,
    vocab,
    MAX_LENGTH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 10. Initialize Model
# ============================================================

model = LightweightCNN(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_filters=NUM_FILTERS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 11. Training Loop
# ============================================================

print("\nTraining CNN model...")

best_val_macro_f1 = -1.0
best_model_path = os.path.join(MODELS_DIR, "cnn_model.pt")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()

    total_train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * input_ids.size(0)

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS}, "
                f"Batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    print("\n" + "-" * 60)
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"Average training loss: {avg_train_loss:.4f}")

    val_result = evaluate_model(
        model=model,
        dataloader=val_loader,
        split_name="validation"
    )

    if val_result["macro_f1"] > best_val_macro_f1:
        best_val_macro_f1 = val_result["macro_f1"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "hyperparameters": {
                    "max_vocab_size": MAX_VOCAB_SIZE,
                    "max_length": MAX_LENGTH,
                    "embed_dim": EMBED_DIM,
                    "num_filters": NUM_FILTERS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                },
                "best_val_macro_f1": best_val_macro_f1,
            },
            best_model_path
        )

        print(f"Saved best CNN model to {best_model_path}")


# ============================================================
# 12. Load Best Model and Evaluate on Test Set
# ============================================================

print("\nLoading best CNN model for final test evaluation...")

checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

final_val_result = evaluate_model(
    model=model,
    dataloader=val_loader,
    split_name="validation"
)

test_result = evaluate_model(
    model=model,
    dataloader=test_loader,
    split_name="test"
)


# ============================================================
# 13. Save Results
# ============================================================

save_predictions(
    texts=val_texts,
    labels=val_labels,
    preds=final_val_result["preds"],
    split_name="validation"
)

save_predictions(
    texts=test_texts,
    labels=test_labels,
    preds=test_result["preds"],
    split_name="test"
)

save_confusion_matrix(
    cm=final_val_result["confusion_matrix"],
    split_name="validation"
)

save_confusion_matrix(
    cm=test_result["confusion_matrix"],
    split_name="test"
)

results_df = pd.DataFrame([
    {
        "model": "Lightweight CNN",
        "split": "validation",
        "loss": final_val_result["loss"],
        "accuracy": final_val_result["accuracy"],
        "macro_f1": final_val_result["macro_f1"],
        "true_negative": final_val_result["true_negative"],
        "false_positive": final_val_result["false_positive"],
        "false_negative": final_val_result["false_negative"],
        "true_positive": final_val_result["true_positive"],
    },
    {
        "model": "Lightweight CNN",
        "split": "test",
        "loss": test_result["loss"],
        "accuracy": test_result["accuracy"],
        "macro_f1": test_result["macro_f1"],
        "true_negative": test_result["true_negative"],
        "false_positive": test_result["false_positive"],
        "false_negative": test_result["false_negative"],
        "true_positive": test_result["true_positive"],
    }
])

results_path = os.path.join(RESULTS_DIR, "cnn_results.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("Final CNN results")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nSaved CNN result summary to {results_path}")


# ============================================================
# 14. Report-Ready Summary
# ============================================================

test_row = results_df[results_df["split"] == "test"].iloc[0]

print("\nReport-ready summary:")
print(
    f"The lightweight CNN achieved {test_row['accuracy']:.4f} accuracy "
    f"and {test_row['macro_f1']:.4f} macro-F1 on the IMDb test set. "
    f"This result will be compared against the TF-IDF Logistic Regression "
    f"baseline and the fine-tuned BERT model."
)