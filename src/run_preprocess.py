import pandas as pd
from pathlib import Path
from src.preprocessor import preprocess
from src.saving_loading.save_preprocessing import save_data, save_preprocessor

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"

def main():
    print("=== Starting Preprocessing Stage ===")
    
    # 1. Load Data
    print("📥 Loading raw data...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    X = raw_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = raw_df['Survived']

    # 2. Get the Transformer Logic
    preprocessor = preprocess()

    # 3. Fit and Transform
    print("⚙️ Fitting preprocessor and transforming data...")
    X_processed = preprocessor.fit_transform(X)
    X_processed_df = pd.DataFrame(X_processed) # Convert to DataFrame for easy CSV saving

    # 4. Save Everything
    save_data(X_processed_df, y)
    save_preprocessor(preprocessor)
    
    print("=== Preprocessing Complete ===")

if __name__ == "__main__":
    main()