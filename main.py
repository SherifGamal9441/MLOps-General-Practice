import os
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from dotenv import load_dotenv

import sklearn.ensemble
import sklearn.linear_model
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from src.saving_loading.save_preprocessing import save_data
from src.preprossesor import preprocess
from src.train import train
from src.evaluate import evaluate_model


# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
X_PATH = PROCESSED_DIR / "X.csv"
Y_PATH = PROCESSED_DIR / "y.csv"
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)

@hydra.main(version_base="1.3", config_path="config", config_name="model")
def main(cfg: DictConfig):
    print("=== Starting ML Orchestration Pipeline ===\n")
    os.chdir(ROOT_DIR)
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    os.environ["MLFLOW_TRACKING_URI"] = cfg.model.tracking_uri

    tracking_uri = cfg.model.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "Titanic_Experiments"
    mlflow.set_experiment(experiment_name)

    client = MlflowClient(tracking_uri=tracking_uri)

    model_name = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)

    param_string = "_".join([f"{k}-{v}" for k, v in model_params.items()])
    artifact_base_name = f"{model_name}_{param_string}"
    print(f"Active Configuration: {artifact_base_name}")

    # --- START MLFLOW RUN ---
    with mlflow.start_run(run_name=model_name):
        print("[MLFLOW] Run started. Logging parameters...")
        print(f"[MLFLOW] Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"[MLFLOW] Artifact URI: {mlflow.get_artifact_uri()}") 
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model_params)

        # --- Data Loading ---
        print("📊 Loading RAW data for the Pipeline...")
        raw_df = pd.read_csv(RAW_DATA_PATH)
        
        # Ensure we only pass the exact raw columns the preprocessor expects
        X = raw_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        y = raw_df['Survived'].values.ravel()

        # --- Model Instantiation ---
        preprocessor_obj = preprocess() 
        
        if hasattr(sklearn.ensemble, model_name):
            model_class = getattr(sklearn.ensemble, model_name)
        elif hasattr(sklearn.linear_model, model_name):
            model_class = getattr(sklearn.linear_model, model_name)
        else:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = model_class(**model_params)

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_obj),
            ('classifier', model)
        ])

        # --- Training ---
        trained_model = train(model_pipeline, X, y, artifact_base_name)

        # --- Evaluation ---
        evaluate_model(trained_model, X, y, artifact_base_name)
    
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
