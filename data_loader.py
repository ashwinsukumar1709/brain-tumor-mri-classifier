# =============================================================================
# data_loader.py — Brain Tumor MRI Classifier
# Author: Ashwin Sukumar
# =============================================================================
# This file handles everything related to loading and preparing images for
# training. It takes raw image files from the disk and converts them into
# numerical arrays (tensors) that the neural network can process.
#
# KEY CONCEPT: Data Augmentation
#   Real-world datasets are limited. To make the model more robust, we
#   artificially create variations of existing images (flip, rotate, zoom)
#   during training. This teaches the model to recognise tumors regardless
#   of the orientation or scale of the MRI scan.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# TensorFlow / Keras imports
# tensorflow.keras.preprocessing.image has tools for loading image datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import all settings from our config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_data_generators():
    """
    Creates and returns three data generators:
      1. train_generator    — augmented training data
      2. val_generator      — validation data (no augmentation)
      3. test_generator     — test data (no augmentation)

    WHAT IS A DATA GENERATOR?
    Instead of loading all 7,000 images into RAM at once (which would crash
    most computers), a generator loads images in small batches on demand.
    It reads a BATCH_SIZE number of images, feeds them to the model, then
    loads the next batch. This is memory-efficient and scalable.

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """

    # -------------------------------------------------------------------------
    # TRAINING DATA GENERATOR — with augmentation
    # -------------------------------------------------------------------------
    # ImageDataGenerator defines how images are loaded and transformed.
    # The augmentation parameters below randomly modify training images
    # to make the model more generalizable.
    train_datagen = ImageDataGenerator(
        # Normalise pixel values from [0, 255] to [0.0, 1.0].
        # Neural networks learn better when inputs are small numbers near 0.
        rescale=1.0 / 255.0,

        # Hold back 20% of training data for validation (no augmentation applied
        # to validation data — it must reflect real, unmodified images).
        validation_split=config.VALIDATION_SPLIT,

        # AUGMENTATION PARAMETERS (only applied to training images):

        # Rotate images randomly up to 20 degrees (clockwise or anticlockwise).
        # MRI scans can be taken at slightly different angles.
        rotation_range=20,

        # Shift images horizontally by up to 10% of the image width.
        # Simulates slightly off-centre scans.
        width_shift_range=0.1,

        # Shift images vertically by up to 10% of the image height.
        height_shift_range=0.1,

        # Zoom into the image by up to 20%.
        # Simulates different zoom levels in MRI machines.
        zoom_range=0.2,

        # Randomly flip images left-right (horizontal mirror).
        # Tumors can appear on either side of the brain.
        horizontal_flip=True,

        # How to fill in empty pixels created by shifts/rotations.
        # "nearest" copies the nearest existing pixel — looks most natural.
        fill_mode="nearest"
    )

    # Validation and test generators: ONLY normalise pixel values (no augmentation).
    # We want to evaluate the model on real, unmodified images.
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # -------------------------------------------------------------------------
    # TRAINING GENERATOR
    # flow_from_directory() scans the folder, assigns class labels based on
    # subfolder names, and yields batches of (images, labels) during training.
    # -------------------------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        directory=config.TRAIN_DIR,        # Folder path containing class subfolders
        target_size=config.IMG_SIZE,        # Resize all images to (224, 224)
        batch_size=config.BATCH_SIZE,       # Load 32 images per batch
        class_mode="categorical",           # One-hot encode labels: [1,0,0,0], [0,1,0,0], etc.
        color_mode="rgb",                   # Load as 3-channel RGB (even if grayscale)
        subset="training",                  # Use the training portion (80%)
        seed=config.RANDOM_SEED,            # For reproducibility
        shuffle=True                        # Shuffle images each epoch
    )

    # -------------------------------------------------------------------------
    # VALIDATION GENERATOR (20% of training data, no augmentation)
    # -------------------------------------------------------------------------
    val_generator = train_datagen.flow_from_directory(
        directory=config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        subset="validation",               # Use the validation portion (20%)
        seed=config.RANDOM_SEED,
        shuffle=False                      # Don't shuffle validation data
    )

    # -------------------------------------------------------------------------
    # TEST GENERATOR (completely separate test set — never seen during training)
    # -------------------------------------------------------------------------
    test_generator = val_test_datagen.flow_from_directory(
        directory=config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False                      # Keep order fixed for accurate evaluation
    )

    # Print a summary so we know how many images were found
    print(f"\n{'='*55}")
    print(f"  Data loaded successfully!")
    print(f"{'='*55}")
    print(f"  Training samples   : {train_generator.samples}")
    print(f"  Validation samples : {val_generator.samples}")
    print(f"  Test samples       : {test_generator.samples}")
    print(f"  Classes found      : {list(train_generator.class_indices.keys())}")
    print(f"  Image size         : {config.IMG_SIZE}")
    print(f"  Batch size         : {config.BATCH_SIZE}")
    print(f"{'='*55}\n")

    return train_generator, val_generator, test_generator


def visualise_samples(train_generator, n_samples=12, save_path=None):
    """
    Displays a grid of sample training images with their class labels.
    Useful to visually confirm the data was loaded correctly.

    Args:
        train_generator: The training data generator
        n_samples (int): Number of sample images to display (default: 12)
        save_path (str): If provided, saves the plot to this file path
    """
    # Reverse the class_indices dict so we can look up: index → class name
    # class_indices looks like: {'glioma': 0, 'meningioma': 1, ...}
    # idx_to_class looks like:  {0: 'glioma', 1: 'meningioma', ...}
    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}

    # Get one batch of images and labels from the generator
    # next() asks the generator to produce the next batch
    images, labels = next(train_generator)

    # Set up a grid of subplots (3 rows × 4 columns = 12 images)
    n_cols = 4
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9))
    fig.suptitle("Sample MRI Images from Training Dataset",
                 fontsize=16, fontweight="bold", y=1.01)

    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            # images[i] is a (224, 224, 3) array with values in [0.0, 1.0]
            # matplotlib imshow() can display it directly
            ax.imshow(images[i])

            # labels[i] is a one-hot vector like [0, 1, 0, 0]
            # np.argmax() finds the index of the highest value (= 1 here → index 1)
            class_idx = np.argmax(labels[i])
            class_name = idx_to_class[class_idx]

            ax.set_title(class_name.upper(), fontsize=11, fontweight="bold",
                         color="#1E3A5F")
            ax.axis("off")  # Hide the x and y axes (not needed for images)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sample images saved → {save_path}")

    plt.show()


def get_class_distribution(generator):
    """
    Counts how many images belong to each class and plots a bar chart.
    Checks for class imbalance — if one class has far more images than others,
    the model might become biased towards it.

    Args:
        generator: A Keras ImageDataGenerator flow object
    """
    # generator.classes is an array of integer labels for every image
    # e.g., [0, 0, 2, 1, 3, 0, 1, ...] where 0=glioma, 1=meningioma, etc.
    class_counts = {}

    # Count images per class using numpy's unique() function
    # unique_classes = [0, 1, 2, 3]
    # counts         = [number of images in each class]
    unique_classes, counts = np.unique(generator.classes, return_counts=True)
    idx_to_class = {v: k for k, v in generator.class_indices.items()}

    print("\nClass Distribution:")
    print("-" * 40)
    for cls_idx, count in zip(unique_classes, counts):
        name = idx_to_class[cls_idx]
        class_counts[name] = count
        # Create a simple text bar chart in the terminal
        bar = "█" * (count // 30)
        print(f"  {name:<15} {count:>5} images  {bar}")
    print("-" * 40)

    # Plot a coloured bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#2563EB", "#16A34A", "#D97706", "#DC2626"]
    bars = ax.bar(list(class_counts.keys()), list(class_counts.values()),
                  color=colours, edgecolor="white", linewidth=1.2)

    # Add count labels on top of each bar
    for bar, count in zip(bars, class_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontweight="bold",
                fontsize=12)

    ax.set_title("Class Distribution in Training Dataset",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tumor Type", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_ylim(0, max(class_counts.values()) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Class distribution chart saved → {save_path}")
    plt.show()

    return class_counts
