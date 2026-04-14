"""
main.py
-------
Entry point for the AI Medical Image Analysis System.
Run this file to execute the complete pipeline:
  setup → train → evaluate → visualize → predict
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import get_data_generators
from model      import build_model, print_model_summary
from train      import train_model
from evaluate   import evaluate_model
from visualize  import plot_training_history, plot_predictions_grid, plot_gradcam
from predict    import predict_single


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir"   : "data/chest_xray",          # Dataset location
    "model_path" : "models/best_model.keras",   # Where to save model
    "output_dir" : "outputs",                   # Where to save plots
    "epochs"     : 20,                          # Max training epochs
    "lr"         : 1e-4                         # Learning rate
}


def run_pipeline():
    """Runs the complete training + evaluation pipeline."""

    print("\n" + "═" * 60)
    print("    🏥 AI-POWERED MEDICAL IMAGE ANALYSIS SYSTEM")
    print("       Pneumonia Detection from Chest X-Rays")
    print("═" * 60 + "\n")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # ── Phase 1: Load Data ────────────────────────────────────────────────────
    print("📂 Phase 1: Loading Dataset...")
    train_gen, val_gen, test_gen = get_data_generators(CONFIG["data_dir"])

    # ── Phase 2: Build Model ──────────────────────────────────────────────────
    print("🏗️  Phase 2: Building MobileNetV2 Model...")
    model = build_model(learning_rate=CONFIG["lr"])
    print_model_summary(model)

    # ── Phase 3: Train ────────────────────────────────────────────────────────
    print("🚀 Phase 3: Training...")
    history = train_model(
        model,
        train_gen,
        val_gen,
        epochs=CONFIG["epochs"],
        model_save_path=CONFIG["model_path"]
    )

    # ── Phase 4: Evaluate ─────────────────────────────────────────────────────
    print("📊 Phase 4: Evaluating on Test Set...")
    metrics = evaluate_model(model, test_gen, CONFIG["output_dir"])

    # ── Phase 5: Visualize ────────────────────────────────────────────────────
    print("🎨 Phase 5: Generating Visualizations...")
    plot_training_history(history, CONFIG["output_dir"])
    plot_predictions_grid(model, test_gen, CONFIG["output_dir"])

    # Grad-CAM: Pick one test image as example
    sample_img_path = None
    pneumonia_dir = os.path.join(CONFIG["data_dir"], "test", "PNEUMONIA")
    if os.path.exists(pneumonia_dir):
        imgs = [f for f in os.listdir(pneumonia_dir) if f.endswith((".jpeg", ".jpg", ".png"))]
        if imgs:
            sample_img_path = os.path.join(pneumonia_dir, imgs[0])
            plot_gradcam(model, sample_img_path, CONFIG["output_dir"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("✅ PIPELINE COMPLETE")
    print(f"   AUC Score  : {metrics['auc']:.4f}")
    print(f"   F1 Score   : {metrics['f1']:.4f}")
    print(f"   Model saved: {CONFIG['model_path']}")
    print(f"   Outputs    : {CONFIG['output_dir']}/")
    print("═" * 60 + "\n")


def run_predict(image_path: str):
    """Runs prediction on a single image."""
    predict_single(CONFIG["model_path"], image_path, show_result=True)

    # Also generate Grad-CAM for the image
    if os.path.exists(CONFIG["model_path"]):
        import tensorflow as tf
        model = tf.keras.models.load_model(CONFIG["model_path"])
        plot_gradcam(model, image_path, CONFIG["output_dir"])


# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AI Medical Image Analysis — Pneumonia Detection"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="'train' runs full pipeline | 'predict' runs on single image"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to X-ray image (required for --mode predict)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_pipeline()

    elif args.mode == "predict":
        if args.image is None:
            print("❌ Please provide --image path for predict mode")
            print("   Example: python main.py --mode predict --image data/chest_xray/test/PNEUMONIA/person1.jpeg")
        else:
            run_predict(args.image)