from src.saving_loading.save_model import save_model
import mlflow.sklearn

def train(model, X, y, artifact_base_name):
    print("Fitting model...")
    model.fit(X, y)

    print("Saving model artifact...")
    # Add the extension here
    full_filename = f"{artifact_base_name}.joblib"
    save_model(model, filename=full_filename)
    try:
        print("[MLFLOW] Logging model to MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("[MLFLOW] log_model succeeded!")
    except Exception as e:
        print(f"[MLFLOW] log_model FAILED: {e}")
        raise  # re-raise so it doesn't silently pass
    
    return model
