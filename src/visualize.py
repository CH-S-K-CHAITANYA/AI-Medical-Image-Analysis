"""
visualize.py
------------
Provides:
  1. Training history plots (accuracy & loss curves)
  2. Sample prediction grid (shows correct/wrong predictions)
  3. Grad-CAM heatmap (shows WHICH part of X-ray triggered AI)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import tensorflow as tf


# ─── 1. TRAINING HISTORY ──────────────────────────────────────────────────────

def plot_training_history(history, output_dir="outputs"):
    """
    Plots accuracy and loss curves from training history.

    Args:
        history    : Keras History object from model.fit()
        output_dir : Where to save the plot
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Pneumonia Detection", fontsize=15, fontweight="bold")

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # ── Accuracy plot ─────────────────────────────────────────────────────────
    axes[0].plot(epochs, history.history["accuracy"],     "b-o", label="Train Accuracy", linewidth=2)
    axes[0].plot(epochs, history.history["val_accuracy"], "r-o", label="Val Accuracy",   linewidth=2)
    axes[0].set_title("Accuracy per Epoch", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # ── Loss plot ─────────────────────────────────────────────────────────────
    axes[1].plot(epochs, history.history["loss"],     "b-o", label="Train Loss", linewidth=2)
    axes[1].plot(epochs, history.history["val_loss"], "r-o", label="Val Loss",   linewidth=2)
    axes[1].set_title("Loss per Epoch", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Training history saved → {save_path}")


# ─── 2. SAMPLE PREDICTION GRID ────────────────────────────────────────────────

def plot_predictions_grid(model, test_gen, output_dir="outputs", num_images=16):
    """
    Shows a 4×4 grid of test images with:
      - True label
      - Predicted label
      - Confidence %
      - Green border = correct | Red border = wrong

    Args:
        model      : Trained Keras model
        test_gen   : Test data generator (shuffle=False)
        output_dir : Save path
        num_images : How many samples to show
    """

    test_gen.reset()

    # Get one batch
    images, labels = next(iter(test_gen))
    preds_prob     = model.predict(images, verbose=0).flatten()
    preds_class    = (preds_prob >= 0.5).astype(int)

    class_names = {0: "NORMAL", 1: "PNEUMONIA"}
    n           = min(num_images, len(images))
    cols        = 4
    rows        = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Sample Predictions — Green: Correct | Red: Wrong",
                 fontsize=14, fontweight="bold", y=1.01)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]

        if i < n:
            img       = images[i]
            true_cls  = int(labels[i])
            pred_cls  = preds_class[i]
            conf      = preds_prob[i] if pred_cls == 1 else 1 - preds_prob[i]
            correct   = (true_cls == pred_cls)

            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"True: {class_names[true_cls]}\n"
                f"Pred: {class_names[pred_cls]} ({conf*100:.1f}%)",
                fontsize=9,
                color="green" if correct else "red",
                fontweight="bold"
            )
            # Border color
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(3)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = f"{output_dir}/predictions_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Predictions grid saved → {save_path}")


# ─── 3. GRAD-CAM HEATMAP ──────────────────────────────────────────────────────

def generate_gradcam(model, image_array, layer_name="out_relu"):
    """
    Generates Gradient-weighted Class Activation Map (Grad-CAM).
    Shows which pixels of the X-ray the AI focused on.

    Args:
        model       : Trained Keras model
        image_array : Single image numpy array (224,224,3) normalized
        layer_name  : Last conv layer name in MobileNetV2

    Returns:
        heatmap (numpy array, same size as input image)
    """

    # Build a model that outputs: last conv layer + final prediction
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(layer_name).output, model.output]
    )

    image_batch = np.expand_dims(image_array, axis=0)   # Add batch dim

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        # For binary: loss is the output probability itself
        loss = predictions[:, 0]

    # Gradients of output w.r.t. conv layer
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv outputs by importance
    conv_outputs  = conv_outputs[0]
    heatmap       = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap       = tf.squeeze(heatmap)

    # Normalize to 0-1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap


def plot_gradcam(model, image_path, output_dir="outputs"):
    """
    Loads an X-ray image, runs Grad-CAM, and saves overlay.

    Args:
        model      : Trained Keras model
        image_path : Path to a single X-ray .jpeg/.png
        output_dir : Save path
    """

    # Load and preprocess image
    img_bgr  = cv2.imread(image_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_224  = cv2.resize(img_rgb, (224, 224))
    img_norm = img_224 / 255.0

    # Generate heatmap
    heatmap = generate_gradcam(model, img_norm)

    # Resize heatmap to image size and colorize
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    overlay = (0.6 * img_224 + 0.4 * heatmap_rgb).astype(np.uint8)

    # Get prediction for this image
    pred_prob  = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0][0]
    pred_label = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"
    confidence = pred_prob if pred_prob >= 0.5 else 1 - pred_prob

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM Analysis | Prediction: {pred_label} ({confidence*100:.1f}%)",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(img_224);          axes[0].set_title("Original X-Ray");     axes[0].axis("off")
    axes[1].imshow(heatmap_resized,
                   cmap="jet");       axes[1].set_title("Grad-CAM Heatmap");   axes[1].axis("off")
    axes[2].imshow(overlay);          axes[2].set_title("Heatmap Overlay");    axes[2].axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04, label="Activation Intensity")

    plt.tight_layout()
    save_path = f"{output_dir}/gradcam_sample.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Grad-CAM saved → {save_path}")
    print(f"   🔍 AI focused on highlighted regions to make this prediction")