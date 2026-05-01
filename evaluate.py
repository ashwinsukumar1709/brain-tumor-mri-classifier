import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import config
from data_loader import get_data_generators

def evaluate():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("Loading Best Model...")
    model_path = os.path.join(config.SAVE_MODEL_DIR, 'best_model.keras')
    
    # This check prevents errors if run before training
    if not os.path.exists(model_path):
        print("Error: Model not found. Run train.py first.")
        return

    model = load_model(model_path)
    _, _, test_gen = get_data_generators()

    print("Predicting...")
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, 
                yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    print("Saved: results/confusion_matrix.png")

if __name__ == '__main__':
    evaluate()