"""
preprocess.py
-------------
Handles all image loading, augmentation, and data pipeline creation.
Uses TensorFlow's ImageDataGenerator for efficient batch loading.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # MobileNetV2 expects 224x224
BATCH_SIZE  = 32
NUM_CLASSES = 1            # Binary: NORMAL=0, PNEUMONIA=1
SEED        = 42


def get_data_generators(data_dir: str):
    """
    Creates train, validation, and test data generators.

    Args:
        data_dir (str): Path to chest_xray/ folder

    Returns:
        train_gen, val_gen, test_gen
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    # ── Training: augmentation applied to avoid overfitting ──────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,          # Normalize pixels 0→1
        rotation_range=15,            # Random rotation ±15°
        width_shift_range=0.1,        # Horizontal shift 10%
        height_shift_range=0.1,       # Vertical shift 10%
        shear_range=0.1,              # Shear transformation
        zoom_range=0.1,               # Random zoom
        horizontal_flip=True,         # Mirror image (X-rays can be mirrored)
        fill_mode="nearest"           # Fill empty pixels with nearest value
    )

    # ── Validation & Test: ONLY rescale, NO augmentation ─────────────────────
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # ── Create generators ─────────────────────────────────────────────────────
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",          # 0 = NORMAL, 1 = PNEUMONIA
        color_mode="rgb",             # MobileNetV2 needs RGB
        shuffle=True,
        seed=SEED
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False
    )

    # ── Print dataset info ────────────────────────────────────────────────────
    print("\n📊 Dataset Summary:")
    print(f"   Train   → {train_gen.samples} images | Classes: {train_gen.class_indices}")
    print(f"   Val     → {val_gen.samples} images")
    print(f"   Test    → {test_gen.samples} images")
    print(f"   Batch   → {BATCH_SIZE} | Image size → {IMG_SIZE}\n")

    return train_gen, val_gen, test_gen