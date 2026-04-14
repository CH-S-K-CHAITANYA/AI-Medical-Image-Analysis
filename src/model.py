"""
model.py
--------
Builds the MobileNetV2 Transfer Learning model.

Architecture:
  MobileNetV2 (frozen base, ImageNet weights)
      ↓
  GlobalAveragePooling2D
      ↓
  Dense(128) + Dropout
      ↓
  Dense(64)  + Dropout
      ↓
  Dense(1, sigmoid) → NORMAL / PNEUMONIA
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2


def build_model(input_shape=(224, 224, 3), learning_rate=1e-4) -> tf.keras.Model:
    """
    Builds MobileNetV2-based binary classifier.

    Args:
        input_shape    : Tuple (H, W, C) — default (224, 224, 3)
        learning_rate  : Adam optimizer LR — default 1e-4

    Returns:
        Compiled Keras model
    """

    # ── Step 1: Load MobileNetV2 pre-trained on ImageNet ─────────────────────
    # include_top=False → removes the original ImageNet classification head
    # weights='imagenet' → uses 1.4M image pre-training for free!
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,            # Remove ImageNet head
        weights="imagenet"            # Use pre-trained weights
    )

    # ── Step 2: Freeze the base model ─────────────────────────────────────────
    # Frozen layers won't update during training.
    # We preserve the powerful ImageNet features already learned.
    base_model.trainable = False
    print(f"✅ MobileNetV2 loaded | Layers: {len(base_model.layers)} | All FROZEN")

    # ── Step 3: Build our custom classification head ───────────────────────────
    inputs = tf.keras.Input(shape=input_shape, name="xray_input")

    # Pass through frozen base
    x = base_model(inputs, training=False)

    # Pool feature maps: (7,7,1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # First Dense block
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)    # Prevents overfitting

    # Second Dense block
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # Output: sigmoid gives probability between 0 and 1
    # < 0.5 → NORMAL, ≥ 0.5 → PNEUMONIA
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="PneumoniaDetector_MobileNetV2")

    # ── Step 4: Compile the model ──────────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",   # Standard loss for binary classification
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),             # Area Under ROC Curve
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def unfreeze_and_finetune(model: tf.keras.Model,
                           fine_tune_from_layer: int = 100,
                           learning_rate: float = 1e-5) -> tf.keras.Model:
    """
    Optional: Unfreeze top layers of MobileNetV2 for fine-tuning.
    Call this AFTER initial training converges.

    Args:
        model              : Already trained model
        fine_tune_from_layer: Layer index from which to unfreeze
        learning_rate      : Much smaller LR for fine-tuning

    Returns:
        Re-compiled model with some layers unfrozen
    """
    base_model = model.layers[1]              # MobileNetV2 is layer index 1
    base_model.trainable = True

    # Freeze layers before fine_tune_from_layer
    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"🔓 Fine-tuning: {unfrozen} layers unfrozen from MobileNetV2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


def print_model_summary(model: tf.keras.Model):
    """Prints trainable parameter count."""
    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n📐 Model: {model.name}")
    print(f"   Total params     : {total:,}")
    print(f"   Trainable params : {trainable:,}")
    print(f"   Frozen params    : {total - trainable:,}\n")