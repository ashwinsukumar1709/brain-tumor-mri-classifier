import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config
from data_loader import get_data_generators
from model import build_model

def train():
    os.makedirs(config.SAVE_MODEL_DIR, exist_ok=True)
    
    train_gen, val_gen, test_gen = get_data_generators()
    model, base_model = build_model()

    checkpoint = ModelCheckpoint(
        os.path.join(config.SAVE_MODEL_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    print("\n--- Starting Training ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        epochs=config.EPOCHS_PHASE1,
        validation_data=val_gen,
        callbacks=[checkpoint]
    )
    print("Training Complete.")

if __name__ == '__main__':
    train()