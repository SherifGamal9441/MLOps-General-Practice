import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import sklearn.ensemble
import sklearn.linear_model
import mlflow
from src.train import train
from src.evaluate import evaluate_model

load_dotenv()

@hydra.main(version_base="1.3", config_path="config", config_name="model")
def main(cfg: DictConfig):
    print("=== Starting ML Orchestration Pipeline ===\n")

    # MLflow Auth & Setup
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(cfg.model.tracking_uri)  # only once
    mlflow.set_experiment("Titanic_Cloud_Production")

    model_name = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    param_string = "_".join([f"{k}-{v}" for k, v in model_params.items()])
    artifact_base_name = f"{model_name}_{param_string}"

    with mlflow.start_run(run_name=model_name):
        # 1. Load Processed Data using Hydra Paths
        print("Loading processed data...")
        X = pd.read_csv(cfg.paths.processed_X)
        y = pd.read_csv(cfg.paths.processed_y).values.ravel()

        # 2. Instantiate Model
        if hasattr(sklearn.ensemble, model_name):
            model_class = getattr(sklearn.ensemble, model_name)
        elif hasattr(sklearn.linear_model, model_name):
            model_class = getattr(sklearn.linear_model, model_name)
        else:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = model_class(**model_params)

        # 3. Train & Evaluate (Passing Hydra paths down)
        trained_model = train(model, X, y, artifact_base_name, cfg.paths.models_dir)
        
        evaluate_model(trained_model, X, y, artifact_base_name, cfg.paths.reports_dir)
    
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()