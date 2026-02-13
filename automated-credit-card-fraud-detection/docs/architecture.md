# Architecture

- Data: CSV samples live in `data/`
- Training: `src/train.py` trains a RandomForest and writes to `models/`
- Prediction: `src/predict.py` loads the model and outputs predictions
- CI: GitHub Actions workflow runs tests
- Containerization: `Dockerfile` builds a container that can train the model
