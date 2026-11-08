
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "Models"
DATA_DIR = ROOT / "Data"
INCOMING_DIR = DATA_DIR / "incoming_images"
LOG_DIR = ROOT / "logs"

MODEL_DIR.mkdir(exist_ok=True)
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def main():
    print("Loading Fashion-MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    print("Preparing image data...")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Building CNN model...")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("Training model...")

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    model_path = MODEL_DIR / "fashion_cnn.keras"
    model.save(model_path)

    print(f"Model saved to: {model_path}")

    logging.info("Model training completed.")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOG_DIR / "training_accuracy.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOG_DIR / "training_loss.png")
    plt.close()

    print("Creating demo images for API testing...")

    if not any(INCOMING_DIR.iterdir()):
        for index in range(20):
            image = Image.fromarray(
                (x_test[index].squeeze() * 255).astype("uint8")
            )
            image.save(INCOMING_DIR / f"demo_{index:02d}.png")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
