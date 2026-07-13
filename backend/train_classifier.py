"""
Train a lightweight CNN classifier on MNIST for automatic sketch digit prediction.

Architecture: Simple 3-layer CNN → 99%+ accuracy on MNIST test set.
Saves model to: weights/mnist_classifier.keras
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_classifier():
    """Build a small but highly accurate CNN for MNIST classification."""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])
    return model


def train():
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add channel dim
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    model = build_classifier()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Train with early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]

    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Save
    save_path = os.path.join("weights", "mnist_classifier.keras")
    os.makedirs("weights", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train()
