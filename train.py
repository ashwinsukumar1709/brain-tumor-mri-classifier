# =============================================================================
# train.py — Brain Tumor MRI Classifier
# Author: Ashwin Sukumar
# =============================================================================
# This is the main training script. Run this file to train the model.
# It handles both Phase 1 (frozen base) and Phase 2 (fine-tuning).
#
# HOW TO RUN:
#   python train.py
#
# WHAT HAPPENS WHEN YOU RUN IT:
#   1. Data is loaded from data/Training/ and data/Testing/
#   2. Phase 1: Model trains for EPOCHS_PHASE1 epochs (only the head layers)
#   3. Phase 2: Last 100 layers of MobileNetV2 are unfrozen and fine-tuned
#   4. The best model is saved to results/brain_tumor_model.h5
#   5. Training plots are saved to results/training_history.png
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,    # Stops training if validation loss stops improving
    ModelCheckpoint,  # Saves the best model during training
    ReduceLROnPlateau # Reduces learning rate when training plateaus
)

from data_loader import create_data_generators
from model import build_model, print_model_summary


def create_callbacks(monitor_metric="val_accuracy"):
    """
    Creates a list of Keras callbacks — special functions that run during
    training to monitor progress, save models, and adjust settings.

    WHAT ARE CALLBACKS?
    Think of them like automatic supervisors that watch training and intervene
    when needed (e.g., stop early if overfitting, save if accuracy improves).

    Args:
        monitor_metric (str): The metric to watch ('val_accuracy' or 'val_loss')

    Returns:
        list: A list of Keras callback objects
    """

    # CALLBACK 1: EarlyStopping
    # Stops training if val_accuracy hasn't improved in 5 consecutive epochs.
    # This prevents wasting time training after the model has stopped learning.
    # restore_best_weights=True rolls back to the best epoch's weights.
    early_stop = EarlyStopping(
        monitor=monitor_metric,
        patience=5,                   # Wait 5 epochs before stopping
        mode="max",                   # "max" because higher accuracy = better
        restore_best_weights=True,    # Roll back to best weights
        verbose=1                     # Print a message when triggered
    )

    # CALLBACK 2: ModelCheckpoint
    # Saves the model to disk only when val_accuracy improves.
    # This ensures we always have the best model saved, even if the model
    # later overfits and accuracy drops.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor=monitor_metric,
        save_best_only=True,          # Only save if this epoch is the best so far
        mode="max",
        verbose=1
    )

    # CALLBACK 3: ReduceLROnPlateau
    # If val_accuracy doesn't improve for 3 epochs, cut the learning rate
    # in half. This helps the model escape flat regions in the loss landscape.
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,                   # New LR = old LR × 0.5
        patience=3,                   # Wait 3 epochs before reducing
        min_lr=1e-7,                  # Never reduce LR below this value
        verbose=1
    )

    return [early_stop, checkpoint, reduce_lr]


def plot_training_history(history1, history2=None, save_path=None):
    """
    Plots training and validation accuracy/loss curves.
    These plots help diagnose:
      - Overfitting: training accuracy >> validation accuracy (bad)
      - Underfitting: both accuracy values are low (needs more epochs)
      - Good fit: both curves are high and close together (ideal)

    Args:
        history1: Keras History object from Phase 1 training
        history2: Keras History object from Phase 2 training (optional)
        save_path (str): Path to save the plot image
    """
    # Combine Phase 1 and Phase 2 histories if both are provided
    if history2 is not None:
        acc     = history1.history["accuracy"]     + history2.history["accuracy"]
        val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
        loss    = history1.history["loss"]         + history2.history["loss"]
        val_loss= history1.history["val_loss"]     + history2.history["val_loss"]
        phase1_end = len(history1.history["accuracy"])  # Where Phase 1 ends
    else:
        acc     = history1.history["accuracy"]
        val_acc = history1.history["val_accuracy"]
        loss    = history1.history["loss"]
        val_loss= history1.history["val_loss"]
        phase1_end = None

    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Training History — Brain Tumor MRI Classifier",
                 fontsize=15, fontweight="bold")

    # ── Plot 1: Accuracy ──
    ax1.plot(epochs, acc,     "b-o", label="Train Accuracy",      markersize=4)
    ax1.plot(epochs, val_acc, "r-o", label="Validation Accuracy", markersize=4)
    if phase1_end:
        ax1.axvline(x=phase1_end, color="grey", linestyle="--", linewidth=1.5,
                    label=f"Phase 2 starts (epoch {phase1_end+1})")
    ax1.set_title("Model Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # ── Plot 2: Loss ──
    ax2.plot(epochs, loss,     "b-o", label="Train Loss",      markersize=4)
    ax2.plot(epochs, val_loss, "r-o", label="Validation Loss", markersize=4)
    if phase1_end:
        ax2.axvline(x=phase1_end, color="grey", linestyle="--", linewidth=1.5,
                    label=f"Phase 2 starts (epoch {phase1_end+1})")
    ax2.set_title("Model Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Training history plot saved → {save_path}")
    plt.show()


def train():
    """
    Main training function. Runs Phase 1 and Phase 2 training sequentially.
    """
    # ── Set random seeds for reproducibility ──
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("=" * 60)
    print("   Brain Tumor MRI Classifier — Training")
    print("=" * 60)

    # ── STEP 1: Load data ──
    print("\n[1/5] Loading and preparing data...")
    train_gen, val_gen, test_gen = create_data_generators()

    # Calculate steps per epoch (how many batches = one full pass through data)
    # steps = ceil(total_samples / batch_size)
    steps_train = int(np.ceil(train_gen.samples / config.BATCH_SIZE))
    steps_val   = int(np.ceil(val_gen.samples   / config.BATCH_SIZE))

    # ── STEP 2: Build Phase 1 model ──
    print("\n[2/5] Building model (Phase 1: frozen base)...")
    model = build_model(phase=1)
    print_model_summary(model)

    # ── STEP 3: Train Phase 1 (only the classification head) ──
    print(f"\n[3/5] Training Phase 1 ({config.EPOCHS_PHASE1} epochs)...")
    print("      Only the Dense + Dropout layers will update in this phase.\n")

    history1 = model.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE1,
        steps_per_epoch=steps_train,
        validation_data=val_gen,
        validation_steps=steps_val,
        callbacks=create_callbacks(),
        verbose=1
    )

    print(f"\n  Phase 1 complete!")
    best_val_acc_p1 = max(history1.history["val_accuracy"])
    print(f"  Best validation accuracy (Phase 1): {best_val_acc_p1:.4f} "
          f"({best_val_acc_p1*100:.1f}%)")

    # ── STEP 4: Fine-tune Phase 2 (unfreeze last 100 layers) ──
    print(f"\n[4/5] Rebuilding model for Phase 2 fine-tuning...")
    print(f"      Loading best Phase 1 weights then unfreezing last "
          f"{config.FINE_TUNE_AT} layers...\n")

    # Rebuild the model with Phase 2 settings (different trainability + LR)
    model_p2 = build_model(phase=2)
    # Load the best weights saved during Phase 1
    model_p2.load_weights(config.MODEL_SAVE_PATH)

    print(f"\n  Training Phase 2 ({config.EPOCHS_PHASE2} epochs)...\n")
    history2 = model_p2.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE2,
        steps_per_epoch=steps_train,
        validation_data=val_gen,
        validation_steps=steps_val,
        callbacks=create_callbacks(),
        verbose=1
    )

    best_val_acc_p2 = max(history2.history["val_accuracy"])
    print(f"\n  Phase 2 complete!")
    print(f"  Best validation accuracy (Phase 2): {best_val_acc_p2:.4f} "
          f"({best_val_acc_p2*100:.1f}%)")

    # ── STEP 5: Plot and save training history ──
    print("\n[5/5] Saving training plots...")
    plot_path = os.path.join(config.RESULTS_DIR, "training_history.png")
    plot_training_history(history1, history2, save_path=plot_path)

    print("\n" + "=" * 60)
    print("   Training Complete!")
    print(f"   Best model saved → {config.MODEL_SAVE_PATH}")
    print("=" * 60)

    return model_p2


if __name__ == "__main__":
    # This block only runs if you execute: python train.py directly
    # It does NOT run if train.py is imported by another script.
    trained_model = train()
