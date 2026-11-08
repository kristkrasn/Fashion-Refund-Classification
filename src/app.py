
import base64
import io
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "Models" / "fashion_cnn.keras"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

app = Flask(__name__)

model = tf.keras.models.load_model(MODEL_PATH)


def decode_image(encoded_image):
    if "," in encoded_image:
        encoded_image = encoded_image.split(",", 1)[1]

    raw = base64.b64decode(encoded_image)

    image = Image.open(io.BytesIO(raw))
    image = image.convert("L")
    image = image.resize((28, 28))

    array = np.asarray(image).astype("float32") / 255.0

    return np.expand_dims(array, axis=(0, -1))


def predict_one(encoded_image):
    image_array = decode_image(encoded_image)

    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probability)
            for name, probability in zip(
                CLASS_NAMES, probabilities
            )
        }
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image_base64" not in data:
            return jsonify({"error": "Missing image_base64"}), 400

        result = predict_one(data["image_base64"])

        logging.info("Single image prediction completed.")

        return jsonify(result)

    except Exception:
        logging.exception("Prediction failed.")
        return jsonify({"error": "Image prediction failed"}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()

        if not data or not isinstance(data.get("images"), list):
            return jsonify({"error": "Expected images list"}), 400

        if len(data["images"]) == 0:
            return jsonify({"error": "Images list is empty"}), 400

        if len(data["images"]) > 100:
            return jsonify({"error": "Maximum batch size is 100"}), 400

        results = []

        for item in data["images"]:
            result = predict_one(item["image_base64"])
            result["filename"] = item.get("filename", "unknown")
            results.append(result)

        logging.info(
            "Batch prediction completed: %s images",
            len(results)
        )

        return jsonify({
            "processed": len(results),
            "results": results
        })

    except Exception:
        logging.exception("Batch prediction failed.")
        return jsonify({"error": "Batch prediction failed"}), 400


if __name__ == "__main__":
   app.run(host="127.0.0.1", port=5055, debug=False)
