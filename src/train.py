"""
train.py
--------
Handles model training with:
  - Early stopping (stops when val_loss stops improving)
  - Model checkpointing (saves best model automatically)
  - Learning rate reduction on plateau
  - Class weight balancing (dataset is imbalanced!)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks


def compute_class_weights(train_gen) -> dict:
    """
    Computes class weights to handle dataset imbalance.
    Pneumonia images >> Normal images in Kaggle dataset.

    Returns:
        dict {0: weight_normal, 1: weight_pneumonia}
    """
    counter = np.bincount(train_gen.classes)          # [count_NORMAL, count_PNEUMONIA]
    total   = sum(counter)
    weights = {
        i: total / (len(counter) * count)
        for i, count in enumerate(counter)
    }
    print(f"⚖️  Class Weights → NORMAL: {weights[0]:.3f} | PNEUMONIA: {weights[1]:.3f}")
    return weights


def get_callbacks(model_save_path: str) -> list:
    """
    Returns list of training callbacks.

    Args:
        model_save_path: Where to save best model (.keras)

    Returns:
        List of Keras callbacks
    """

    # ── Callback 1: Save best model ───────────────────────────────────────────
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_auc",            # Save when val AUC improves
        mode="max",
        save_best_only=True,
        verbose=1
    )

    # ── Callback 2: Stop if no improvement ───────────────────────────────────
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,                   # Wait 5 epochs before stopping
        restore_best_weights=True,
        verbose=1
    )

    # ── Callback 3: Reduce LR when stuck ──────────────────────────────────────
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,                   # Halve the learning rate
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # ── Callback 4: Log training to CSV ───────────────────────────────────────
    csv_logger = callbacks.CSVLogger("outputs/training_log.csv", append=False)

    return [checkpoint, early_stop, reduce_lr, csv_logger]


def train_model(model, train_gen, val_gen, epochs=20, model_save_path="models/best_model.keras"):
    """
    Trains the model.

    Args:
        model          : Compiled Keras model
        train_gen      : Training data generator
        val_gen        : Validation data generator
        epochs         : Max training epochs
        model_save_path: Path to save best model

    Returns:
        history object (contains accuracy/loss per epoch)
    """

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    class_weights = compute_class_weights(train_gen)
    cbs           = get_callbacks(model_save_path)

    print(f"\n🚀 Starting Training | Epochs: {epochs} | Batch: {train_gen.batch_size}")
    print("─" * 60)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1
    )

    print("\n✅ Training Complete!")
    return history