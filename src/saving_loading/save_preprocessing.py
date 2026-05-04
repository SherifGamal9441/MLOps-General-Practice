import joblib
import pandas as pd
from pathlib import Path

def save_data(X, y, x_path, y_path):
    # Ensure the directories exist dynamically
    Path(x_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("💾 Saving processed datasets...")
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)
    print(f"✅ Data saved to: {Path(x_path).parent}")

    return None

def save_preprocessor(preprocessor, preprocessor_path):
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("📦 Saving fitted preprocessor artifact...")
    joblib.save(preprocessor, preprocessor_path)
    print(f"✅ Preprocessor saved to: {preprocessor_path}")

    return None