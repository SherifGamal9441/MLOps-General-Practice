import joblib
from pathlib import Path

# Setup paths dynamically (assuming save_model.py is inside the 'src' folder)
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

def save_pipeline(pipeline_object, filename="titanic_pipeline.joblib"):
    """
    Serializes and saves the trained machine learning pipeline.
    """
    # Ensure the artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Construct the full file path
    save_path = ARTIFACTS_DIR / filename
    
    # Dump the model
    joblib.dump(pipeline_object, save_path)
    print(f"Artifact successfully saved to: {save_path}")
    
    return save_path