import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from src.utils import load_data, save_model


def main(data_path: str, output: str):
    df = load_data(data_path)
    X = df[[c for c in df.columns if c not in ("label",)]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_model(clf, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "data/sample.csv"))
    parser.add_argument("--output", default=os.getenv("MODEL_PATH", "models/model.joblib"))
    args = parser.parse_args()
    main(args.data, args.output)
