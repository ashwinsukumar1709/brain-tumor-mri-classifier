# =============================================================================
# predict.py — Brain Tumor MRI Classifier
# Author: Ashwin Sukumar
# =============================================================================
# Use this script to classify a single MRI image using the trained model.
#
# HOW TO RUN:
#   python predict.py --image path/to/your/mri_scan.jpg
#
# EXAMPLE:
#   python predict.py --image data/Testing/glioma/Te-gl_0010.jpg
# =============================================================================

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image


def load_and_preprocess_image(img_path):
    """
    Loads a single image from disk and prepares it for the model.

    Steps:
      1. Load image from file → PIL Image object
      2. Resize to (224, 224) — the size MobileNetV2 expects
      3. Convert to numpy array of shape (224, 224, 3)
      4. Normalise pixel values from [0, 255] → [0.0, 1.0]
      5. Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
         because Keras models always expect a BATCH of images

    Args:
        img_path (str): Path to the image file

    Returns:
        img_array (np.array): Shape (1, 224, 224, 3), ready for model.predict()
        img_display (np.array): Shape (224, 224, 3), for displaying
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load image and resize to target size
    img = keras_image.load_img(img_path, target_size=config.IMG_SIZE,
                               color_mode="rgb")

    # Convert PIL Image to numpy array of shape (224, 224, 3)
    img_array = keras_image.img_to_array(img)

    # Save a copy for display (values still in [0, 255])
    img_display = img_array.astype("uint8")

    # Normalise to [0.0, 1.0] — same preprocessing used during training
    img_array = img_array / 255.0

    # np.expand_dims adds a batch dimension at axis 0
    # Shape: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img_display


def predict_single_image(img_path, model=None):
    """
    Predicts the tumor class of a single MRI image.

    Args:
        img_path (str): Path to the MRI image file
        model: Loaded Keras model (loads from disk if not provided)

    Returns:
        dict: {
            'predicted_class': str,
            'confidence': float (0-100),
            'all_probabilities': dict {class_name: probability}
        }
    """
    # Load model if not provided
    if model is None:
        if not os.path.exists(config.MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"No trained model found at {config.MODEL_SAVE_PATH}.\n"
                "Please run train.py first."
            )
        print(f"Loading model from {config.MODEL_SAVE_PATH}...")
        model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # Preprocess the image
    img_array, img_display = load_and_preprocess_image(img_path)

    # Run the forward pass — get probability predictions
    # predictions shape: (1, 4) — one row, four class probabilities
    predictions = model.predict(img_array, verbose=0)

    # predictions[0] removes the batch dimension → shape: (4,)
    probs = predictions[0]

    # np.argmax() returns the index of the highest probability
    predicted_idx   = np.argmax(probs)
    predicted_class = config.CLASS_NAMES[predicted_idx]
    confidence      = float(probs[predicted_idx]) * 100

    # Build a dict of all class probabilities
    all_probs = {cls: float(probs[i]) * 100
                 for i, cls in enumerate(config.CLASS_NAMES)}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "img_display": img_display,
        "img_path": img_path
    }


def visualise_prediction(result):
    """
    Creates a two-panel visualisation:
      Left: The MRI image with prediction overlay
      Right: Horizontal bar chart of all class probabilities

    Args:
        result (dict): Output from predict_single_image()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Brain Tumor MRI — Classification Result",
                 fontsize=15, fontweight="bold")

    # ── LEFT PANEL: MRI image ──
    ax1.imshow(result["img_display"])
    ax1.axis("off")

    # Determine colour based on class
    class_colours = {
        "glioma":     "#DC2626",
        "meningioma": "#D97706",
        "notumor":    "#16A34A",
        "pituitary":  "#2563EB"
    }
    box_colour = class_colours.get(result["predicted_class"], "#374151")

    # Add a coloured prediction box at the bottom of the image
    ax1.text(
        0.5, 0.04,
        f"PREDICTION: {result['predicted_class'].upper()}\n"
        f"Confidence: {result['confidence']:.1f}%",
        transform=ax1.transAxes,
        fontsize=12, fontweight="bold", color="white",
        ha="center", va="bottom",
        bbox=dict(facecolor=box_colour, alpha=0.85, boxstyle="round,pad=0.5")
    )
    ax1.set_title(f"Input MRI: {os.path.basename(result['img_path'])}",
                  fontsize=11)

    # ── RIGHT PANEL: Probability bar chart ──
    classes  = list(result["all_probabilities"].keys())
    probs    = list(result["all_probabilities"].values())
    colours  = [class_colours.get(c, "#374151") for c in classes]

    # Highlight the predicted class bar
    bar_alpha = [1.0 if c == result["predicted_class"] else 0.35 for c in classes]
    bars = ax2.barh(classes, probs, color=colours, alpha=1.0, edgecolor="white")
    for bar, alpha in zip(bars, bar_alpha):
        bar.set_alpha(alpha)

    # Add percentage labels on each bar
    for bar, prob in zip(bars, probs):
        ax2.text(
            min(prob + 1.5, 95), bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%",
            va="center", ha="left", fontsize=11, fontweight="bold"
        )

    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Probability (%)", fontsize=12)
    ax2.set_title("Class Probability Distribution", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)
    ax2.tick_params(axis="y", labelsize=11)

    plt.tight_layout()

    # Save and show
    save_path = os.path.join(
        config.RESULTS_DIR,
        f"prediction_{os.path.splitext(os.path.basename(result['img_path']))[0]}.png"
    )
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Prediction visualisation saved → {save_path}")
    plt.show()

    return save_path


def main():
    """Parses command line arguments and runs prediction."""
    parser = argparse.ArgumentParser(
        description="Classify a brain MRI image as glioma, meningioma, "
                    "no tumor, or pituitary tumor."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input MRI image file (JPEG or PNG)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the visualisation plot (useful for batch scripting)"
    )
    args = parser.parse_args()

    # Run prediction
    result = predict_single_image(args.image)

    # Print results to terminal
    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Image          : {os.path.basename(result['img_path'])}")
    print(f"  Prediction     : {result['predicted_class'].upper()}")
    print(f"  Confidence     : {result['confidence']:.2f}%")
    print("\n  All probabilities:")
    for cls, prob in sorted(result["all_probabilities"].items(),
                            key=lambda x: -x[1]):
        bar = "▓" * int(prob / 5)
        print(f"    {cls:<15} {prob:>6.2f}%  {bar}")
    print("=" * 50)

    if not args.no_plot:
        visualise_prediction(result)


if __name__ == "__main__":
    main()
