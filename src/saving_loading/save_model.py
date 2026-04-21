import joblib
from pathlib import Path

# Setup paths dynamically (assuming save_model.py is inside the 'src/saving_loading' folder)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def save_model(model, filename="titanic_model.joblib"):
    """
    Serializes and saves the trained machine learning model.
    """
    # Ensure the artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Construct the full file path
    save_path = ARTIFACTS_DIR / filename

    # Dump the model
    joblib.dump(model, save_path)
    print(f"Model successfully saved to: {save_path}")

    return save_path
