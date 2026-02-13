import os
from src.train import main


def test_train_runs(tmp_path):
    # Use sample data and temporary model output
    data = "data/sample.csv"
    out = tmp_path / "model.joblib"
    main(data, str(out))
    assert out.exists()
