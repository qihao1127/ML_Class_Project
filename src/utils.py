import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def evaluate_predictions(y_true, y_pred, model_name, split_name, print_report=True):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    if print_report:
        print("\n" + "=" * 60)
        print(f"{model_name} results on {split_name} set")
        print("=" * 60)
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["negative", "positive"],
                digits=4,
                zero_division=0,
            )
        )

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "true_negative": cm[0, 0],
        "false_positive": cm[0, 1],
        "false_negative": cm[1, 0],
        "true_positive": cm[1, 1],
        "confusion_matrix": cm,
    }


def save_predictions(texts, labels, preds, output_path):
    pred_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": labels,
            "predicted_label": preds,
            "correct": np.array(labels) == np.array(preds),
        }
    )
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def save_confusion_matrix(cm, model_name, split_name, output_path):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {output_path}")


def safe_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("=", "")
        .replace(".", "")
    )