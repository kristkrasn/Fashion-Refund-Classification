
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "Models" / "fashion_cnn.keras"
RESULTS_DIR = ROOT / "Data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test dataset...")
    (_, _), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    print("Evaluating model...")
    probabilities = model.predict(x_test, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print(f"Test accuracy: {accuracy:.4f}")

    with open(
        RESULTS_DIR / "classification_report.txt",
        "w"
    ) as file:
        file.write(
            classification_report(
                y_test,
                predictions,
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(
            report["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report["macro avg"]["f1-score"]
        )
    }

    with open(
        RESULTS_DIR / "metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - CNN Model")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()

    print("Evaluation completed.")
    print(metrics)


if __name__ == "__main__":
    main()
