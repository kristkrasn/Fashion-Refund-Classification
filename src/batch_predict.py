
import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "Data" / "incoming_images"
RESULTS_DIR = ROOT / "Data" / "results"

API_URL = "http://127.0.0.1:5055/predict/batch"

BATCH_SIZE = 16

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        path for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        print("No images found.")
        return

    all_results = []

    print(f"Found {len(image_paths)} images.")

    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start:start + BATCH_SIZE]

        images = [
            {
                "filename": path.name,
                "image_base64": encode_image(path)
            }
            for path in batch_paths
        ]

        response = requests.post(
            API_URL,
            json={"images": images},
            timeout=120
        )

        response.raise_for_status()

        batch_results = response.json()["results"]
        all_results.extend(batch_results)

        print(
            f"Processed {len(all_results)} "
            f"of {len(image_paths)} images."
        )

    dataframe = pd.DataFrame(all_results)

    dataframe["probabilities"] = dataframe[
        "probabilities"
    ].apply(str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        RESULTS_DIR / f"batch_predictions_{timestamp}.csv"
    )

    dataframe.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
