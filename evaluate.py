# =============================================================================
# evaluate.py — Brain Tumor MRI Classifier
# Author: Ashwin Sukumar
# =============================================================================
# This file evaluates the trained model on the test set and produces
# publication-quality visualisations:
#   1. Confusion Matrix — shows where the model makes mistakes
#   2. Classification Report — precision, recall, F1-score per class
#   3. ROC Curves — performance across different decision thresholds
#   4. Sample Predictions — visual grid of correct/incorrect predictions
#
# HOW TO RUN (after train.py has completed):
#   python evaluate.py
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,        # Count of TP, FP, FN, TN per class
    classification_report,   # Precision, recall, F1 per class
    roc_curve,               # ROC curve data points
    auc                      # Area Under the Curve (AUC) score
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from data_loader import create_data_generators


def load_model():
    """Loads the saved trained model from disk."""
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No saved model found at {config.MODEL_SAVE_PATH}.\n"
            f"Please run train.py first."
        )
    print(f"  Loading model from: {config.MODEL_SAVE_PATH}")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print(f"  Model loaded successfully.\n")
    return model


def get_predictions(model, test_generator):
    """
    Runs the model on every image in the test set and collects predictions.

    Args:
        model: The loaded Keras model
        test_generator: The test data generator

    Returns:
        y_true (np.array): True class indices, shape (n_samples,)
        y_pred (np.array): Predicted class indices, shape (n_samples,)
        y_prob (np.array): Predicted probabilities, shape (n_samples, 4)
    """
    print("  Running predictions on test set...")

    # Reset the generator so it starts from the first image
    test_generator.reset()

    # model.predict() runs the forward pass for all batches
    # It returns the softmax probabilities for each image
    # Shape: (n_test_images, 4) — e.g., [[0.7, 0.1, 0.1, 0.1], ...]
    y_prob = model.predict(
        test_generator,
        steps=int(np.ceil(test_generator.samples / config.BATCH_SIZE)),
        verbose=1
    )

    # np.argmax(axis=1) finds the class index with the highest probability
    # for each image. This is the model's final prediction.
    # e.g., [0.7, 0.1, 0.1, 0.1] → argmax → 0 (glioma)
    y_pred = np.argmax(y_prob, axis=1)

    # test_generator.classes gives the true label for every image
    y_true = test_generator.classes

    print(f"  Total test images: {len(y_true)}")
    accuracy = np.mean(y_pred == y_true)
    print(f"  Overall test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")

    return y_true, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plots a normalised confusion matrix.

    WHAT IS A CONFUSION MATRIX?
    A grid where:
      - Rows = actual class
      - Columns = predicted class
      - Diagonal = correct predictions (higher is better)
      - Off-diagonal = mistakes (lower is better)

    Normalised version shows percentages (fraction of true class).

    Args:
        y_true: Array of true class indices
        y_pred: Array of predicted class indices
        save_path (str): Path to save the figure
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalise: divide each row by its total to get percentages
    cm_normalised = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — Brain Tumor MRI Classifier",
                 fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_normalised],
        ["Counts", "Normalised (%)"],
        ["d", ".2f"]
    ):
        sns.heatmap(
            data,
            annot=True,                    # Show numbers in cells
            fmt=fmt,                       # Format: integers or decimals
            cmap="Blues",                  # Blue colour scale
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label",      fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved → {save_path}")
    plt.show()


def plot_roc_curves(y_true, y_prob, save_path=None):
    """
    Plots ROC (Receiver Operating Characteristic) curves for each class.

    WHAT IS A ROC CURVE?
    It plots True Positive Rate vs False Positive Rate at different
    classification thresholds. The AUC (Area Under Curve) summarises
    performance: AUC = 1.0 is perfect, AUC = 0.5 is random guessing.

    We use a ONE-vs-REST approach: for each class, we treat it as a binary
    problem (this class vs all others).

    Args:
        y_true: True class indices (integers)
        y_prob: Predicted probabilities (shape: n_samples × 4)
        save_path: Path to save the figure
    """
    # Convert integer labels to binary (one-hot) format for ROC calculation
    # e.g., class 0 → [1, 0, 0, 0], class 2 → [0, 0, 1, 0]
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))

    colours = ["#2563EB", "#16A34A", "#D97706", "#DC2626"]
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (class_name, colour) in enumerate(zip(config.CLASS_NAMES, colours)):
        # Compute ROC curve for class i vs all others
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot the curve
        ax.plot(fpr, tpr, colour, linewidth=2.5,
                label=f"{class_name.capitalize()} (AUC = {roc_auc:.3f})")

    # Plot the diagonal (random classifier baseline)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier (AUC = 0.5)")

    ax.set_title("ROC Curves — One-vs-Rest", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ROC curves saved → {save_path}")
    plt.show()


def plot_sample_predictions(model, test_generator, n=16, save_path=None):
    """
    Shows a grid of sample test images with predicted and true labels.
    Correct predictions are shown in green, incorrect in red.

    Args:
        model: The trained Keras model
        test_generator: Test data generator
        n (int): Number of sample images to show
        save_path: Path to save the figure
    """
    test_generator.reset()

    # Collect one batch of images and their true labels
    images, labels_onehot = next(test_generator)
    y_true_batch = np.argmax(labels_onehot, axis=1)

    # Get model predictions for this batch
    y_prob_batch = model.predict(images, verbose=0)
    y_pred_batch = np.argmax(y_prob_batch, axis=1)

    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
    n = min(n, len(images))

    n_cols = 4
    n_rows = n // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 11))
    fig.suptitle("Sample Predictions — Green = Correct | Red = Incorrect",
                 fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i])
            true_name = idx_to_class[y_true_batch[i]]
            pred_name = idx_to_class[y_pred_batch[i]]
            confidence = y_prob_batch[i][y_pred_batch[i]] * 100

            # Colour the title green if correct, red if wrong
            colour = "#16A34A" if y_true_batch[i] == y_pred_batch[i] else "#DC2626"
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} ({confidence:.0f}%)",
                fontsize=9, color=colour, fontweight="bold"
            )
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sample predictions saved → {save_path}")
    plt.show()


def evaluate():
    """Main evaluation function — runs all visualisations."""
    print("=" * 60)
    print("   Brain Tumor MRI Classifier — Evaluation")
    print("=" * 60)

    # Load model and data
    model = load_model()
    _, _, test_gen = create_data_generators()

    # Get predictions
    y_true, y_pred, y_prob = get_predictions(model, test_gen)

    # Print detailed classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=[c.capitalize() for c in config.CLASS_NAMES],
        digits=4
    ))

    # Save all visualisations
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Generating visualisations...\n")

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_true, y_prob,
        save_path=os.path.join(config.RESULTS_DIR, "roc_curves.png")
    )
    plot_sample_predictions(
        model, test_gen,
        save_path=os.path.join(config.RESULTS_DIR, "sample_predictions.png")
    )

    print("\n" + "=" * 60)
    print("   Evaluation Complete! All figures saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
