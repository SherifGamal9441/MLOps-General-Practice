import os
import pandas as pd
import mlflow
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv, find_dotenv

load_dotenv()

@hydra.main(version_base="1.3", config_path="../config", config_name="model")
def main(cfg: DictConfig):
    # 1. Credentials & Configuration
    dagshub_user = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    tracking_uri = cfg.model.tracking_uri
    repo_name = cfg.model.repo_name

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(tracking_uri)

    print("Connecting to DagsHub MLflow Registry...")
    
    # 2. Modern MLflow fetching using the @ ALIAS syntax
    model_name = "Titanic_Survival_Model"
    alias = "Production"
    model_uri = f"models:/{model_name}@{alias}"
    
    print(f"Downloading '{model_name}' (Alias: {alias}) from {repo_name}...")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully!")
        
        print("Preparing test passenger data...")
        test_passengers = pd.DataFrame({
            'Pclass': [3, 1],               
            'Sex': [1, 0],                  
            'Age': [22.0, 38.0],            
            'SibSp': [1, 1],               
            'Parch': [0, 0],               
            'Fare': [7.25, 71.28],         
            'Embarked': [2, 0]              
        })
        
        predictions = model.predict(test_passengers)
        
        print("\n🔮 PREDICTION RESULTS:")
        for i, pred in enumerate(predictions):
            status = "Survived" if pred == 1 else "Did Not Survive"
            print(f"Passenger {i+1}: {status}")
            
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    main()