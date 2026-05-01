# =============================================================================
# config.py — Brain Tumor MRI Classifier
# Author: Ashwin Sukumar
# =============================================================================
# This file holds ALL the settings for the project in one place.
# Instead of spreading numbers throughout the code, we define them here.
# If you want to change the image size, learning rate, or number of epochs,
# you only need to edit THIS file — not hunt through every script.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# PATHS — Where to find the data and where to save results
# -----------------------------------------------------------------------------

# Base directory of this project (wherever this file sits)
# os.path.dirname(__file__) returns the folder where config.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset folder (you will download this from Kaggle)
# Expected structure:
#   data/
#     Training/
#       glioma/        ← MRI images of glioma tumors
#       meningioma/    ← MRI images of meningioma tumors
#       notumor/       ← MRI images with no tumor
#       pituitary/     ← MRI images of pituitary tumors
#     Testing/
#       glioma/
#       meningioma/
#       notumor/
#       pituitary/
TRAIN_DIR = os.path.join(BASE_DIR, "data", "Training")
TEST_DIR  = os.path.join(BASE_DIR, "data", "Testing")

# Where to save the trained model weights (.h5 file)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results", "brain_tumor_model.h5")

# Where to save evaluation plots (confusion matrix, accuracy curves, etc.)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------------------------------
# IMAGE SETTINGS
# -----------------------------------------------------------------------------

# MobileNetV2 (the pretrained model we use) was designed for 224x224 images.
# We resize ALL input images to this size so they match what the network expects.
# Think of it like resizing a photo to fit a specific picture frame.
IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)  # A tuple (height, width) — used by Keras

# Number of colour channels:
#   3 = RGB (red, green, blue) — standard colour images
#   MRI images are grayscale but we convert them to 3 channels to match
#   the input format that MobileNetV2 was trained on
CHANNELS = 3

# -----------------------------------------------------------------------------
# TRAINING SETTINGS
# -----------------------------------------------------------------------------

# BATCH SIZE — How many images the model looks at in one step before updating.
# Larger batch = faster but uses more memory. 32 is a safe default.
BATCH_SIZE = 32

# EPOCHS — How many full passes through the entire training dataset.
# Phase 1 (training only the new top layers we added)
EPOCHS_PHASE1 = 10
# Phase 2 (fine-tuning — we "unfreeze" the last part of MobileNetV2 too)
EPOCHS_PHASE2 = 10

# LEARNING RATE — Controls how big each learning "step" is.
# Too large = the model overshoots and doesn't learn properly.
# Too small = the model learns extremely slowly.
# 0.001 is a common starting point for Phase 1.
# 0.0001 (ten times smaller) is used for Phase 2 fine-tuning to be careful.
LR_PHASE1 = 0.001
LR_PHASE2 = 0.0001

# VALIDATION SPLIT — During training, we hold back this fraction of training
# data to check how the model is doing on unseen examples.
# 0.2 means 20% of training data is used for validation.
VALIDATION_SPLIT = 0.2

# RANDOM SEED — Setting this ensures we get the same random results every time
# we run the code. Makes the project reproducible (important in research!).
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# CLASS LABELS — The 4 categories of brain tumors we classify
# -----------------------------------------------------------------------------
# This list MUST match the folder names in your data/Training/ directory.
# The order matters — it determines how the model assigns numbers to classes.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Number of classes (automatically counted from CLASS_NAMES)
NUM_CLASSES = len(CLASS_NAMES)  # = 4

# -----------------------------------------------------------------------------
# MODEL SETTINGS
# -----------------------------------------------------------------------------

# DROPOUT RATE — During training, we randomly "turn off" this fraction of
# neurons to prevent the model from memorising training data (overfitting).
# 0.5 means 50% of neurons are randomly disabled during each training step.
DROPOUT_RATE = 0.5

# Number of neurons in the dense (fully connected) layer we add on top of
# MobileNetV2. 256 is a good balance of capacity vs speed.
DENSE_UNITS = 256

# How many layers from the END of MobileNetV2 we "unfreeze" in Phase 2.
# Unfreezing allows those layers to update their weights and adapt to MRI data.
FINE_TUNE_AT = 100  # Unfreeze the last 100 layers of MobileNetV2
