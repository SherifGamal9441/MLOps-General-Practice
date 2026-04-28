import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# 1. Import the modules themselves, not the specific classes
import sklearn.ensemble
import sklearn.linear_model

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

@hydra.main(version_base="1.3", config_path="config", config_name="model")
def main(cfg: DictConfig):
    print("=== Starting ML Orchestration Pipeline ===\n")

    model_name = cfg.model.name
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)

    param_string = "_".join([f"{k}-{v}" for k, v in model_params.items()])
    artifact_base_name = f"{model_name}_{param_string}"
    print(f"Active Configuration: {artifact_base_name}")

    # --- 2. Data Loading ---
    if X_PATH.exists() and Y_PATH.exists():
        print("\n[DATA] Processed features found. Loading from disk...")
        X = pd.read_csv(X_PATH)
        y = pd.read_csv(Y_PATH)
    else:
        print("\n[DATA] Processed features NOT found. Loading raw data...")
        raw_df = pd.read_csv(RAW_DATA_PATH)

        print("[PREPROCESSING] Executing transformation pipeline...")
        X_raw = raw_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        y = raw_df["Survived"]

        preprocessor_obj = preprocess()
        X_processed_array = preprocessor_obj.fit_transform(X_raw)

        feature_names = preprocessor_obj.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed_array, columns=feature_names)

        save_data(X_processed, y, X_filename="X.csv", y_filename="y.csv")
        X = X_processed

    # --- 3. Dynamic Model Instantiation ---
    print(f"\n[INIT] Dynamically loading {model_name}...")

    # Check which module contains the requested model
    if hasattr(sklearn.ensemble, model_name):
        model_class = getattr(sklearn.ensemble, model_name)
    elif hasattr(sklearn.linear_model, model_name):
        model_class = getattr(sklearn.linear_model, model_name)
    else:
        # Failsafe if you type a model name wrong in the YAML
        raise ValueError(
            f"Model '{model_name}' not found in sklearn.ensemble or sklearn.linear_model."
        )

    # Instantiate the model by passing the dictionary as kwargs
    model = model_class(**model_params)

    # --- 4. Training ---
    print("\n[TRAINING] Starting Model Training...")
    trained_model = train(model, X, y, artifact_base_name)

    # --- 5. Evaluation ---
    print("\n[EVALUATION] Generating Metrics and Visualizations...")
    evaluate_model(trained_model, X, y, artifact_base_name)

    print("\n=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()
