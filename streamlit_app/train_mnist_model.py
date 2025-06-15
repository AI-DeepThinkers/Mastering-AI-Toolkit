# streamlit_app/train_mnist_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import os

def train_and_save_mnist_model():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    print("🧠 Training MNIST CNN model...")
    model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=2)

    # Save model in project root
    model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn_model.h5")
    model.save(model_path)
    print(f"✅ Model saved at: {model_path}")

if __name__ == "__main__":
    train_and_save_mnist_model()
