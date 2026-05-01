# =============================================================================
# model.py — Brain Tumor MRI Classifier
# Author: Ashwin Sukumar
# =============================================================================
# This file defines the neural network architecture.
#
# APPROACH: Transfer Learning with MobileNetV2
# ─────────────────────────────────────────────
# Training a deep neural network from scratch requires millions of images and
# weeks of compute time. Instead, we use TRANSFER LEARNING:
#
# 1. We take MobileNetV2 — a powerful CNN pretrained on ImageNet (1.4 million
#    images, 1000 classes). It has already learned to detect edges, textures,
#    shapes, and complex visual patterns.
#
# 2. We FREEZE those learned weights (stop them from changing) so they act as
#    a universal feature extractor for our MRI images.
#
# 3. We add our own small classification layers ON TOP and train only those
#    new layers using our ~7,000 MRI images.
#
# 4. In Phase 2, we UNFREEZE the last part of MobileNetV2 and fine-tune it
#    on MRI data with a very small learning rate.
#
# WHY MobileNetV2?
#   • Lightweight — runs on a laptop CPU (no GPU required)
#   • High accuracy — designed for medical/mobile imaging tasks
#   • Well-documented — easy to explain in professor emails and interviews
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,   # Pools spatial features into a single vector
    Dense,                    # Fully connected layer
    Dropout,                  # Regularisation — randomly drops neurons
    BatchNormalization        # Normalises activations — speeds up training
)
from tensorflow.keras.optimizers import Adam


def build_model(phase=1):
    """
    Builds and compiles the full classification model.

    ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────┐
    │  INPUT: (224, 224, 3) MRI image              │
    ├──────────────────────────────────────────────┤
    │  MobileNetV2 Base (154 layers)               │
    │  Pretrained on ImageNet                      │
    │  Phase 1: ALL frozen (no weight updates)     │
    │  Phase 2: Last 100 layers unfrozen           │
    ├──────────────────────────────────────────────┤
    │  GlobalAveragePooling2D                      │
    │  Converts (7, 7, 1280) → (1280,) vector      │
    ├──────────────────────────────────────────────┤
    │  BatchNormalization                          │
    ├──────────────────────────────────────────────┤
    │  Dense(256, activation='relu')               │
    │  Dropout(0.5)                                │
    ├──────────────────────────────────────────────┤
    │  Dense(4, activation='softmax')              │
    │  OUTPUT: probabilities for 4 classes         │
    └──────────────────────────────────────────────┘

    Args:
        phase (int): 1 = freeze base model, 2 = unfreeze last 100 layers

    Returns:
        model: A compiled Keras Model ready for training
    """

    # -------------------------------------------------------------------------
    # STEP 1: Load the MobileNetV2 base model (pretrained on ImageNet)
    # -------------------------------------------------------------------------
    # include_top=False means we EXCLUDE the original ImageNet classification
    # head (which would give 1000 output classes). We'll add our own head.
    # input_shape tells it our images are (224, 224, 3).
    base_model = MobileNetV2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS),
        include_top=False,
        weights="imagenet"   # Load weights pretrained on ImageNet
    )

    # -------------------------------------------------------------------------
    # STEP 2: Freeze/unfreeze layers based on the training phase
    # -------------------------------------------------------------------------
    if phase == 1:
        # PHASE 1: Freeze ALL base model layers.
        # trainable=False means the layer's weights will NOT change during
        # backpropagation. We treat MobileNetV2 as a fixed feature extractor.
        base_model.trainable = False
        print(f"\n  Phase 1: Base model FROZEN — training only classification head")

    elif phase == 2:
        # PHASE 2: Unfreeze the last `FINE_TUNE_AT` layers.
        # First, make the whole base model trainable...
        base_model.trainable = True
        # ...then freeze everything EXCEPT the last FINE_TUNE_AT layers.
        # This is called fine-tuning: we carefully update the final layers
        # of MobileNetV2 to adapt to MRI-specific features.
        for layer in base_model.layers[:-config.FINE_TUNE_AT]:
            layer.trainable = False
        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        print(f"\n  Phase 2: Fine-tuning last {config.FINE_TUNE_AT} layers of base model")
        print(f"  ({trainable_count} / {len(base_model.layers)} base layers are trainable)")

    # -------------------------------------------------------------------------
    # STEP 3: Build the classification head
    # -------------------------------------------------------------------------
    # We take the OUTPUT of the base model (a 7×7×1280 feature map)
    # and pass it through our custom layers to get 4-class predictions.

    # Get the output tensor from the base model
    x = base_model.output

    # GlobalAveragePooling2D: Reduces (7, 7, 1280) → (1280,)
    # It averages each of the 1280 feature maps into a single number.
    # This is more efficient than Flatten() and less prone to overfitting.
    x = GlobalAveragePooling2D()(x)

    # BatchNormalization: Normalises the activations to have mean≈0, std≈1.
    # This stabilises training and often improves accuracy.
    x = BatchNormalization()(x)

    # Dense layer: A fully connected layer with 256 neurons.
    # ReLU activation: f(x) = max(0, x) — introduces non-linearity.
    # Without non-linearity, stacking layers would be pointless (just linear).
    x = Dense(config.DENSE_UNITS, activation="relu",
               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    # Dropout: During each training step, randomly set 50% of neuron outputs
    # to zero. This prevents the model from relying too heavily on any one
    # neuron (overfitting). Dropped neurons change randomly each step.
    x = Dropout(config.DROPOUT_RATE)(x)

    # OUTPUT LAYER: Dense with 4 neurons (one per tumor class).
    # Softmax activation: Converts raw scores (logits) into probabilities
    # that sum to 1.0. e.g., [0.70, 0.15, 0.10, 0.05] → 70% glioma.
    outputs = Dense(config.NUM_CLASSES, activation="softmax")(x)

    # -------------------------------------------------------------------------
    # STEP 4: Create and compile the full model
    # -------------------------------------------------------------------------
    # Model() stitches together: input → base_model → our custom head → output
    model = Model(inputs=base_model.input, outputs=outputs)

    # Choose learning rate based on phase
    lr = config.LR_PHASE1 if phase == 1 else config.LR_PHASE2

    # COMPILE — Configures the model for training.
    # optimizer: Adam — adaptive learning rate, works well for most problems
    # loss: categorical_crossentropy — standard for multi-class classification
    # metrics: accuracy — percentage of correctly classified images
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def print_model_summary(model):
    """
    Prints a readable summary of the model architecture and parameter counts.
    Helps you understand the model size and identify bottlenecks.
    """
    model.summary()

    # Count total parameters and how many are trainable
    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)

    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Frozen parameters    : {total_params - trainable_params:,}")
    print()
