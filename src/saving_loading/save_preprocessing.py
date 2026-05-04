from pathlib import Path
import joblib

# Setup paths dynamically (assuming this file is in 'src')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PREPROCESSORS_DIR = ROOT_DIR / "artifacts" / "preprocessors"


def save_data(X, y, X_filename="X.csv", y_filename="y.csv"):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    x_path = PROCESSED_DATA_DIR / X_filename
    y_path = PROCESSED_DATA_DIR / y_filename

    print("💾 Saving processed datasets...")
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)
    print(f"✅ Data saved to: {PROCESSED_DATA_DIR}")

    return None

def save_preprocessor(preprocessor, filename="preprocessor.joblib"):
    PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact_path = PREPROCESSORS_DIR / filename
    print("📦 Saving fitted preprocessor artifact...")
    joblib.save(preprocessor, artifact_path)
    print(f"✅ Preprocessor saved to: {artifact_path}")

    return None