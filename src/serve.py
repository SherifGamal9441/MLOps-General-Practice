import os
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from omegaconf import OmegaConf

# 1. Define the Data Schema
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

class BatchRequest(BaseModel):
    passengers: List[Passenger]

# Global dictionary to hold our loaded models
ml_components = {}

# 2. Define the Startup Logic (Runs once when server starts)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting server up. Loading ML components...")
    load_dotenv()
    
    cfg = OmegaConf.load("config/paths.yaml")
    
    # Setup MLflow Auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Load Preprocessor
    preprocessor_path = cfg.paths.preprocessors_dir + "/preprocessor.joblib"
    print("Loading preprocessor from DVC tracking...")
    ml_components["preprocessor"] = joblib.load(preprocessor_path)

    # Load Model using MLflow Registry - latest version
    model_name = "Titanic_Production_Model"
    client = MlflowClient()
    versions = client.get_latest_versions(model_name)
    if not versions:
        raise ValueError(f"No registered versions found for model: {model_name}")

    latest_version = versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"

    print(f"Downloading '{model_name}' version {latest_version} from MLflow Registry...")
    ml_components["model"] = mlflow.pyfunc.load_model(model_uri)
    
    print("Server ready to accept requests.")
    yield
    
    # Cleanup on shutdown
    ml_components.clear()
    print("Server shutting down.")

# 3. Initialize FastAPI
app = FastAPI(title="Titanic Prediction API", lifespan=lifespan)

# 4. Define the Prediction Endpoint
@app.post("/predict")
async def predict_batch(request: BatchRequest):
    try:
        # Convert the list of Pydantic models into a list of dictionaries, then to a DataFrame
        data = [passenger.model_dump() for passenger in request.passengers]
        df = pd.DataFrame(data)

        # Retrieve loaded components
        preprocessor = ml_components["preprocessor"]
        model = ml_components["model"]

        # Preprocess
        X_processed = preprocessor.transform(df)
        X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

        # Predict
        predictions = model.predict(X_processed_df)

        # Return results paired with the original input index
        results = [{"record_index": i, "survived_prediction": int(pred)} for i, pred in enumerate(predictions)]
        
        return {"status": "success", "predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))