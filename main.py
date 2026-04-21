import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Imports mapped to your custom directory structure
from src.saving_loading.save_preprocessing import save_data
from src.preprossesor import preprocess
from src.train import train
from src.evaluate import evaluate_model

# Setup absolute paths dynamically based on main.py's location
ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
X_PATH = PROCESSED_DIR / "X.csv"
Y_PATH = PROCESSED_DIR / "y.csv"

def main():
    print("=== Starting ML Orchestration Pipeline ===\n")

    if X_PATH.exists() and Y_PATH.exists():
        print("[DATA] Processed features found. Loading from disk...")
        X = pd.read_csv(X_PATH)
        y = pd.read_csv(Y_PATH)
    else:
        print("[DATA] Processed features NOT found. Loading raw data...")
        # Load raw data
        try:
            raw_df = pd.read_csv(RAW_DATA_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}. Please ensure the dataset is downloaded.")
        
        print("[PREPROCESSING] Executing transformation pipeline...")
        
        # 1. Isolate the features and target from the raw dataset
        X_raw = raw_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        y = raw_df['Survived']
        
        # 2. Initialize the preprocessor and transform the data
        # Note: calling preprocess() because it returns the ColumnTransformer object
        preprocessor_obj = preprocess() 
        X_processed_array = preprocessor_obj.fit_transform(X_raw)
        
        # 3. Convert the numpy array back to a DataFrame for clean CSV saving
        feature_names = preprocessor_obj.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed_array, columns=feature_names)
        
        # 4. Execute your standalone save logic
        save_data(X_processed, y, X_filename="X.csv", y_filename="y.csv")
        
        # 5. Assign the variables for the training step 
        # (Since it's already in memory, we skip the pd.read_csv step to save time)
        X = X_processed

    # ---------------------------------------------------------
    # STEP 3: Model Definition
    # ---------------------------------------------------------
    print("\n[INIT] Initializing Machine Learning Model...")
    # You define the model here, making it trivial to swap algorithms later
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    # ---------------------------------------------------------
    # STEP 4: Training
    # ---------------------------------------------------------
    print("\n[TRAINING] Starting Model Training...")
    # Assuming your train.py is updated to accept X and y directly from the orchestrator
    trained_model = train(model, X, y)

    # ---------------------------------------------------------
    # STEP 5: Evaluation
    # ---------------------------------------------------------
    print("\n[EVALUATION] Generating Metrics and Visualizations...")
    evaluate_model(trained_model, X, y)
    
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()