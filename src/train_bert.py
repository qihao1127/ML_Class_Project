import os
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

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

MODEL_NAME = "distilbert-base-uncased"

# Setup for speed
USE_SUBSET = True
SUBSET_TRAIN_SIZE = 2000
SUBSET_VAL_SIZE = 500
SUBSET_TEST_SIZE = 1000

MAX_LENGTH = 128
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
set_seed(SEED)
make_dirs(RESULTS_DIR, FIGURES_DIR, MODELS_DIR)

print("CUDA available:", torch.cuda.is_available())


# =========================
# 1. Load IMDb Dataset
# =========================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Shuffle first to avoid taking only one class
train_data = dataset["train"].shuffle(seed=SEED)
test_data = dataset["test"].shuffle(seed=SEED)

if USE_SUBSET:
    train_subset = train_data.select(range(SUBSET_TRAIN_SIZE + SUBSET_VAL_SIZE))
    test_subset = test_data.select(range(SUBSET_TEST_SIZE))
else:
    train_subset = train_data
    test_subset = test_data

# Convert to list[str] explicitly for tokenizer
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

test_texts = [
    str(x) for x in test_subset["text"]
]
test_labels = list(test_subset["label"])

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

print("Train label counts:", np.bincount(train_labels))
print("Validation label counts:", np.bincount(val_labels))
print("Test label counts:", np.bincount(test_labels))


# =========================
# 2. Tokenization
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


train_encodings = tokenize_texts(train_texts)
val_encodings = tokenize_texts(val_texts)
test_encodings = tokenize_texts(test_texts)


# =========================
# 3. Dataset Class
# =========================

class IMDbBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )
        return item


train_dataset = IMDbBERTDataset(train_encodings, train_labels)
val_dataset = IMDbBERTDataset(val_encodings, val_labels)
test_dataset = IMDbBERTDataset(test_encodings, test_labels)


# =========================
# 4. Model
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# =========================
# 5. Metrics for Trainer
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    result = evaluate_predictions(
        y_true=labels,
        y_pred=preds,
        model_name="DistilBERT",
        split_name="validation",
        print_report=False,
    )

    return {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
    }


# =========================
# 6. Training
# =========================

output_dir = os.path.join(MODELS_DIR, "distilbert_imdb")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=20,
    save_strategy="no",
    eval_strategy="epoch",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Training DistilBERT...")
trainer.train()

print("Saving DistilBERT model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)


# =========================
# 7. Final Evaluation
# =========================

all_results = []

for split_name, texts, labels, eval_dataset in [
    ("validation", val_texts, val_labels, val_dataset),
    ("test", test_texts, test_labels, test_dataset),
]:
    print("\n" + "=" * 80)
    print(f"Evaluating DistilBERT on {split_name} set")
    print("=" * 80)

    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    preds = np.argmax(logits, axis=1)

    result = evaluate_predictions(
        y_true=labels,
        y_pred=preds,
        model_name="DistilBERT",
        split_name=split_name,
    )

    all_results.append(
        {
            "model": "DistilBERT",
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
            f"bert_predictions_distilbert_fine_tuned_subset_2000_{split_name}.csv",
        ),
    )

    save_confusion_matrix(
        cm=result["confusion_matrix"],
        model_name="DistilBERT",
        split_name=split_name,
        output_path=os.path.join(
            FIGURES_DIR,
            f"bert_confusion_matrix_distilbert_fine_tuned_subset_2000_{split_name}.png",
        ),
    )


# =========================
# 8. Save Summary Results
# =========================

results_df = pd.DataFrame(all_results)
if USE_SUBSET:
    results_path = os.path.join(
        RESULTS_DIR,
        "bert_results_subset_2000.csv",
    )
else:
    results_path = os.path.join(
        RESULTS_DIR,
        "bert_results.csv",
    )
results_df.to_csv(results_path, index=False)

print("\nBERT results:")
print(results_df)
print(f"\nSaved BERT results to {results_path}")