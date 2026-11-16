
# Fashion Refund Classification

## Project Overview

This project presents an image classification
prototype for a sustainable online fashion retailer.

The system classifies clothing images into ten
categories using a Convolutional Neural Network (CNN).

## Dataset

Fashion-MNIST is used as a benchmark dataset.
It contains 70,000 grayscale images across ten classes.

The dataset is used for prototyping.

## Technologies

- Python 3.12
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flask REST API
- Pillow
- Requests

## Project Structure

- `src/train_model.py` - model training
- `src/evaluate_model.py` - model evaluation
- `src/app.py` - REST API
- `src/batch_predict.py` - batch predictions
- `nightly_batch.sh` - scheduled batch script
- `Data/results/` - prediction and evaluation results
- `Models/` - saved trained model
- `logs/` - application logs

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python src/train_model.py
```

## Evaluation

```bash
python src/evaluate_model.py
```

## Starting the API

```bash
python src/app.py
```

Health endpoint:

`GET /health`

Prediction endpoint:

`POST /predict`

Batch prediction endpoint:

`POST /predict/batch`

## Batch Processing

Place images in `Data/incoming_images/`.

Start the API, then run:

```bash
python src/batch_predict.py
```

Results are saved as CSV files in `Data/results/`.

## Nightly Automation

This project includes a Bash script that automates nightly batch data processing.

The script performs the following actions:
- Activates the Python virtual environment.
- Checks the availability of the REST API.
- Starts the API if it is not running.
- Waits until the API is ready for operation.
- Performs batch image classification.
- Saves prediction results to CSV files.
- Records execution details in log files.

The script is scheduled to run daily at 02:00 using the `cron` scheduler on the macOS operating system.

To run the script manually:

```bash
./nightly_batch.sh
```

## Limitations

Fashion-MNIST contains small grayscale benchmark
images. Performance on real-world customer photographs
may be different.

The prototype requires further validation using
real product-return images.
