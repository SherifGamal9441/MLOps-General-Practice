import joblib
from pathlib import Path

def save_model(model, save_path):
    """
    Serializes and saves the trained machine learning model.
    """
    # Ensure the parent directory (artifacts/models) exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Dump the model
    joblib.dump(model, save_path)
    print(f" Model successfully saved locally to: {save_path}")

    return save_path