from pathlib import Path
from src.saving_loading.save_model import save_model
import mlflow.sklearn

def train(model, X, y, artifact_base_name, models_dir):
    print("Fitting model...")
    model.fit(X, y)

    # Construct the full path using the Hydra directory
    full_filename = f"{artifact_base_name}.joblib"
    save_path = Path(models_dir) / full_filename
    
    # Save locally using the refactored helper
    save_model(model, save_path=save_path)
    
    try:
        print("[MLFLOW] Logging model to MLflow cloud...")
        mlflow.sklearn.log_model(model, "model")
        print("[MLFLOW] log_model succeeded!")
    except Exception as e:
        print(f"[MLFLOW] log_model FAILED: {e}")
        raise 
    
    return model