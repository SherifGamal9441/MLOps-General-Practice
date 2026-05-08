from pathlib import Path
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.saving_loading.save_model import save_model

def train(model, X, y, artifact_base_name, models_dir):

    if all(str(c).isdigit() for c in X.columns):
        raise ValueError(
            f"X has no real feature names — columns are {list(X.columns)}. "
            "Check that processed_X.csv was saved with header=True."
        )


    print("Fitting model...")
    model.fit(X, y)

    # 1. Save locally for DVC tracking
    full_filename = f"{artifact_base_name}.joblib"
    save_path = Path(models_dir) / full_filename
    save_model(model, save_path)

    # 2. Push to DagsHub MLflow Registry
    try:
        print("[MLFLOW] Pushing model to MLflow Registry...")

        # Infer signature from training data — critical for inference endpoints
        predictions = model.predict(X)
        signature = infer_signature(X, predictions)

        # A small input example so MLflow knows the exact payload shape
        input_example = X.iloc[:5]

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="Titanic_Production_Model",
            signature=signature,
            input_example=input_example,
            pip_requirements=["scikit-learn", "pandas", "numpy"],
        )
        
        print("[MLFLOW] log_model succeeded!")
    except Exception as e:
        print(f"[MLFLOW] log_model FAILED: {e}")
        raise

    return model