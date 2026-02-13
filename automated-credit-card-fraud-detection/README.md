# Automated Credit Card Fraud Detection

A starter MLOps project scaffold for detecting credit card fraud — includes sample data, a training script, prediction utilities, Dockerfile and CI.

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Train a model with the sample data:

```bash
python src/train.py --data data/sample.csv --output models/model.joblib
```

Run a prediction:

```bash
python src/predict.py --model models/model.joblib --input data/sample.csv
```
