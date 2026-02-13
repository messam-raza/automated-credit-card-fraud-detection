import argparse
import pandas as pd
from src.utils import load_model, load_data


def predict(model_path: str, input_path: str):
    model = load_model(model_path)
    df = load_data(input_path)
    X = df[[c for c in df.columns if c not in ("label",)]]
    preds = model.predict(X)
    print("Predictions:\n", preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--input", default="data/sample.csv")
    args = parser.parse_args()
    predict(args.model, args.input)
