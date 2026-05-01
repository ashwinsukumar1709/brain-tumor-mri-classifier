import os

# Paths
DATA_DIR = 'data/'
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Testing')
SAVE_MODEL_DIR = 'saved_model'
RESULTS_DIR = 'results'

# Model Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5  # Reduced for faster training for beginners
EPOCHS_PHASE2 = 5
LEARNING_RATE = 1e-4

# Class Names
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)