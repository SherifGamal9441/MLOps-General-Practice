import pandas as pd
from pathlib import Path

# Setup paths dynamically (assuming this file is in 'src')
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

def save_data(X, y, X_filename="X_processed.csv", y_filename="y.csv"):
    # Define paths
    x_path = PROCESSED_DATA_DIR / X_filename
    y_path = PROCESSED_DATA_DIR / y_filename
    
    print("Saving processed datasets...")
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)
    
    print(f"Features saved to: {x_path}")
    print(f"Target saved to: {y_path}")
    
    return None