# MLOps-General-Practice

## How to Run

1. `uv run dvc repro` -> Run the preprocessing, training, and evaluation.
2. `uv run python inference.py` -> Test the trained model pulled from the MLflow registry.
3. `uv run python src/serve.py` -> Start the prediction endpoint. Test with Bruno using `POST http://localhost:8000/predict` and pass the passenger data in JSON format in the body.
4. **Batch Prediction (MotherDuck):** Add your token to `.env`, run `uv run python src/seed_motherduck.py` to upload the test data, then run `uv run python src/batch_predict.py` to predict and store results in the database.