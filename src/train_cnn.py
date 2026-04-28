import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from utils import (
    set_seed,
    make_dirs,
    evaluate_predictions,
    save_predictions,
    save_confusion_matrix,
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

MAX_VOCAB_SIZE = 30000
MAX_LENGTH = 256
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

EMBED_DIM = 128
NUM_FILTERS = 128
KERNEL_SIZE = 5
DROPOUT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
make_dirs(RESULTS_DIR, FIGURES_DIR, MODELS_DIR)

print(f"Using device: {DEVICE}")


# =========================
# 1. Text Processing
# =========================

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

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    if len(ids) < max_length:
        ids += [vocab["<pad>"]] * (max_length - len(ids))
    else:
        ids = ids[:max_length]

    return ids


# =========================
# 2. Dataset
# =========================

class IMDbCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.encoded_texts = [
            encode_text(text, vocab, max_length)
            for text in texts
        ]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(
                self.encoded_texts[index],
                dtype=torch.long,
            ),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


# =========================
# 3. Model
# =========================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_size, dropout):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.max(x, dim=2).values
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# =========================
# 4. Train / Evaluate Helpers
# =========================

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(model, dataloader):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# =========================
# 5. Load Data
# =========================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Shuffle first to avoid class-order issue
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
# 6. Build Vocab and Dataloaders
# =========================

print("Building vocabulary...")
vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)

train_dataset = IMDbCNNDataset(
    train_texts,
    train_labels,
    vocab,
    MAX_LENGTH,
)

val_dataset = IMDbCNNDataset(
    val_texts,
    val_labels,
    vocab,
    MAX_LENGTH,
)

test_dataset = IMDbCNNDataset(
    test_texts,
    test_labels,
    vocab,
    MAX_LENGTH,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =========================
# 7. Train CNN
# =========================

model = TextCNN(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_filters=NUM_FILTERS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

best_val_macro_f1 = -1.0

if USE_SUBSET:
    best_model_path = os.path.join(
        MODELS_DIR,
        "best_cnn_model_subset_2000.pt",
    )
else:
    best_model_path = os.path.join(
        MODELS_DIR,
        "best_cnn_model.pt",
    )

patience = 2
epochs_without_improvement = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print("\n" + "=" * 80)
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print("=" * 80)

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
    )
    print(f"Training loss: {train_loss:.4f}")

    val_labels_np, val_preds = predict(model, val_loader)

    val_result = evaluate_predictions(
        y_true=val_labels_np,
        y_pred=val_preds,
        model_name="CNN",
        split_name="validation",
    )

    if val_result["macro_f1"] > best_val_macro_f1:
        best_val_macro_f1 = val_result["macro_f1"]
        epochs_without_improvement = 0

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
                    "use_subset": USE_SUBSET,
                    "subset_train_size": SUBSET_TRAIN_SIZE,
                    "subset_val_size": SUBSET_VAL_SIZE,
                    "subset_test_size": SUBSET_TEST_SIZE,
                },
                "best_val_macro_f1": best_val_macro_f1,
            },
            best_model_path,
        )

        print(f"Saved best CNN model to {best_model_path}")

    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break


# =========================
# 8. Final Evaluation
# =========================

print("\nLoading best CNN model...")
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

all_results = []

for split_name, texts, labels, loader in [
    ("validation", val_texts, val_labels, val_loader),
    ("test", test_texts, test_labels, test_loader),
]:
    y_true, y_pred = predict(model, loader)

    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        model_name="CNN",
        split_name=split_name,
    )

    all_results.append(
        {
            "model": "CNN",
            "split": split_name,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "true_negative": result["true_negative"],
            "false_positive": result["false_positive"],
            "false_negative": result["false_negative"],
            "true_positive": result["true_positive"],
        }
    )

    if USE_SUBSET:
        prediction_filename = f"cnn_predictions_subset_2000_{split_name}.csv"
        figure_filename = f"cnn_confusion_matrix_subset_2000_{split_name}.png"
    else:
        prediction_filename = f"cnn_predictions_{split_name}.csv"
        figure_filename = f"cnn_confusion_matrix_{split_name}.png"

    save_predictions(
        texts=texts,
        labels=labels,
        preds=y_pred,
        output_path=os.path.join(
            RESULTS_DIR,
            prediction_filename,
        ),
    )

    save_confusion_matrix(
        cm=result["confusion_matrix"],
        model_name="CNN",
        split_name=split_name,
        output_path=os.path.join(
            FIGURES_DIR,
            figure_filename,
        ),
    )


# =========================
# 9. Save Summary Results
# =========================

results_df = pd.DataFrame(all_results)

if USE_SUBSET:
    results_path = os.path.join(
        RESULTS_DIR,
        "cnn_results_subset_2000.csv",
    )
else:
    results_path = os.path.join(
        RESULTS_DIR,
        "cnn_results.csv",
    )

results_df.to_csv(results_path, index=False)

print("\nCNN results:")
print(results_df)
print(f"\nSaved CNN results to {results_path}")