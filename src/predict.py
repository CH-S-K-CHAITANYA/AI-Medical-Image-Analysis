"""
predict.py
----------
Loads a saved model and predicts on a single X-ray image.
Used for demo purposes — simulates real clinical usage.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def load_model(model_path: str) -> tf.keras.Model:
    """
    Loads a saved Keras model.

    Args:
        model_path: Path to .keras file

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from → {model_path}")
    return model


def preprocess_single_image(image_path: str) -> tuple:
    """
    Loads and preprocesses a single X-ray image for prediction.

    Args:
        image_path: Path to .jpeg/.png X-ray image

    Returns:
        (img_display, img_normalized_batch)
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    img_bgr  = cv2.imread(image_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_224  = cv2.resize(img_rgb, (224, 224))
    img_norm = img_224 / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)   # Shape: (1, 224, 224, 3)

    return img_224, img_batch


def predict_single(model_path: str, image_path: str, show_result: bool = True) -> dict:
    """
    Full prediction pipeline for a single X-ray image.

    Args:
        model_path  : Path to saved .keras model
        image_path  : Path to X-ray image
        show_result : Whether to display the result plot

    Returns:
        dict with prediction info
    """

    model                = load_model(model_path)
    img_display, img_in  = preprocess_single_image(image_path)

    # Run prediction
    raw_prob   = model.predict(img_in, verbose=0)[0][0]
    label      = "PNEUMONIA" if raw_prob >= 0.5 else "NORMAL"
    confidence = raw_prob if raw_prob >= 0.5 else 1 - raw_prob

    result = {
        "image_path" : image_path,
        "prediction" : label,
        "confidence" : f"{confidence * 100:.2f}%",
        "raw_prob"   : float(raw_prob)
    }

    # ── CLI Output ────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("       🏥 AI PNEUMONIA DETECTION RESULT")
    print("═" * 50)
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {'🔴 ' if label == 'PNEUMONIA' else '🟢 '}{label}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Raw Prob   : {raw_prob:.4f}")
    print("═" * 50)

    if label == "PNEUMONIA":
        print("  ⚠️  AI suggests signs of PNEUMONIA detected.")
        print("  📌 Recommend: Consult a radiologist for confirmation.")
    else:
        print("  ✅ AI suggests lung appears NORMAL.")
        print("  📌 Recommend: Routine follow-up as needed.")
    print("═" * 50 + "\n")

    # ── Visual Output ─────────────────────────────────────────────────────────
    if show_result:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_display)
        color = "red" if label == "PNEUMONIA" else "green"
        ax.set_title(
            f"Prediction: {label}\nConfidence: {result['confidence']}",
            fontsize=14, fontweight="bold", color=color, pad=12
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return result