import os
import hydra
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@hydra.main(version_base="1.3", config_path="config", config_name="model")
def main(cfg: DictConfig):
    # 1. Setup MLflow Authentication
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(cfg.model.tracking_uri)

    # 2. Load Preprocessor via Hydra Paths
    preprocessor_path = Path(cfg.paths.preprocessors_dir) / "preprocessor.joblib"
    print("Loading preprocessor...")
    preprocessor = joblib.load(preprocessor_path)

    # 3. Dynamically get latest registered version
    client = MlflowClient()
    model_name = "Titanic_Production_Model"

    versions = client.get_latest_versions(model_name)
    if not versions:
        raise ValueError(f"No registered versions found for model: {model_name}")

    latest_version = versions[0].version
    print(f"Found model version: {latest_version}")

    # 4. Load model via registry
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")

    # 5. Execute Inference
    sample = pd.DataFrame([{
        "Pclass": 3, "Sex": "male", "Age": 22.0,
        "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"
    }])

    X_processed = preprocessor.transform(sample)
    X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

    prediction = model.predict(X_processed_df)
    print(f"\nPrediction: {prediction}")

if __name__ == "__main__":
    main()