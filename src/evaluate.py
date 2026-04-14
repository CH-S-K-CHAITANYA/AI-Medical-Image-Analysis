"""
evaluate.py
-----------
Evaluates model on test data and generates:
  - Accuracy, Precision, Recall, F1, AUC
  - Confusion matrix (saved as PNG)
  - Classification report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)


def evaluate_model(model, test_gen, output_dir="outputs"):
    """
    Full evaluation pipeline on test set.

    Args:
        model      : Trained Keras model
        test_gen   : Test data generator (shuffle=False!)
        output_dir : Where to save output images
    """

    print("\n📊 Evaluating on Test Set...")
    print("─" * 60)

    # ── Get predictions ───────────────────────────────────────────────────────
    # Reset generator to ensure consistent ordering
    test_gen.reset()

    # Get raw probabilities (0.0 to 1.0)
    y_prob = model.predict(test_gen, verbose=1).flatten()

    # Convert probability to binary class (threshold = 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # True labels from generator
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())  # ['NORMAL', 'PNEUMONIA']

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc    = roc_auc_score(y_true, y_prob)
    f1     = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"\n🎯 Test Results:")
    print(f"   AUC Score : {auc:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"\n📋 Classification Report:")
    print(report)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    plot_roc_curve(y_true, y_prob, auc, output_dir)

    return {"auc": auc, "f1": f1, "y_pred": y_pred, "y_prob": y_prob, "y_true": y_true}


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Generates and saves confusion matrix heatmap."""

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1,
        linecolor="black"
    )
    ax.set_title("Confusion Matrix — Pneumonia Detection", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Add TN/FP/FN/TP labels
    ax.text(0.25, -0.08, "TN", transform=ax.transAxes, fontsize=10, color="green")
    ax.text(0.75, -0.08, "FP", transform=ax.transAxes, fontsize=10, color="red")
    ax.text(0.25,  1.02, "FN", transform=ax.transAxes, fontsize=10, color="orange")
    ax.text(0.75,  1.02, "TP", transform=ax.transAxes, fontsize=10, color="green")

    plt.tight_layout()
    save_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true, y_prob, auc, output_dir):
    """Generates and saves ROC curve."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/roc_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ ROC curve saved → {save_path}")