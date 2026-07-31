"""
Member 2 (AI Architect) pipeline.

Defines the 1D-CNN architecture described in the proposal (3-4 conv layers,
max-pooling + dropout, 2 dense layers, 5-way softmax), compiles it with the
Adam optimizer and categorical cross-entropy loss, handles class imbalance
via computed class weights, and trains/monitors validation metrics per epoch.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CLASSES = ['N', 'S', 'V', 'F', 'Q']
NUM_CLASSES = len(CLASSES)


def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    return X[..., np.newaxis], y  # add channel dim for Conv1D


def build_model(input_len=180, num_classes=NUM_CLASSES):
    model = keras.Sequential([
        layers.Input(shape=(input_len, 1)),

        layers.Conv1D(16, kernel_size=7, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(32, kernel_size=11, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(32, kernel_size=21, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(64, kernel_size=31, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(16, activation='relu'),

        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.arange(NUM_CLASSES), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights (to counter class imbalance):", class_weight_dict)

    model = build_model(input_len=X_train.shape[1])
    model.summary()

    total_params = model.count_params()
    print(f"Total trainable parameters: {total_params}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(os.path.join(RESULTS_DIR, "ecg_1dcnn_model.keras"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    with open(os.path.join(RESULTS_DIR, "model_info.json"), "w") as f:
        json.dump({
            "total_params": int(total_params),
            "epochs_run": len(history.history['loss']),
            "class_weights": class_weight_dict,
        }, f, indent=2)

    print("Model and training history saved to results/")
    return model, history


if __name__ == "__main__":
    train()
