import os
import duckdb
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from prefect import task, flow
from omegaconf import OmegaConf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")

@task(name="Extract Data from MotherDuck", retries=2)
def extract_data() -> pd.DataFrame:
    print("Extracting raw test data...")
    con = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")
    df = con.execute("SELECT * FROM raw_titanic_test").df()
    return df

@task(name="Load ML Components")
def load_ml_components():
    print("Loading preprocessor and model...")
    cfg = OmegaConf.load("config/paths.yaml")
    
    # 1. Load Preprocessor
    preprocessor_path = cfg.paths.preprocessors_dir + "/preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path)

    # 2. Setup MLflow & Load Model
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model_name = "Titanic_Production_Model"
    client = MlflowClient()
    versions = client.get_latest_versions(model_name)
    if not versions:
        raise ValueError(f"No registered versions found for model: {model_name}")
    
    latest_version = versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    return preprocessor, model

@task(name="Transform and Predict")
def predict(df: pd.DataFrame, preprocessor, model) -> pd.DataFrame:
    print("Transforming data and generating predictions...")
    
    # Keep PassengerId aside for the final output
    passenger_ids = df['PassengerId'].copy()
    
    # The test set might have missing columns or extra columns like PassengerId. 
    # We must slice only the features the preprocessor expects.
    features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    X_processed = preprocessor.transform(features)
    X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
    
    predictions = model.predict(X_processed_df)
    
    # Create the final output DataFrame matching Kaggle submission format
    results_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int)
    })
    return results_df

@task(name="Load Predictions to MotherDuck")
def load_to_warehouse(results_df: pd.DataFrame):
    print("Writing predictions back to MotherDuck...")
    con = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")
    
    # DuckDB can write a pandas DataFrame directly to a new table
    con.execute("CREATE TABLE IF NOT EXISTS titanic_predictions AS SELECT * FROM results_df")
    
    # If the table already existed, you might want to insert/append instead:
    # con.execute("INSERT INTO titanic_predictions SELECT * FROM results_df")
    
    print("Batch load complete.")

@flow(name="Titanic Batch Inference Pipeline")
def titanic_batch_job():
    # 1. Extract
    raw_data = extract_data()
    
    # 2. Load ML Assets
    preprocessor, model = load_ml_components()
    
    # 3. Predict
    predictions = predict(raw_data, preprocessor, model)
    
    # 4. Load Output
    load_to_warehouse(predictions)

if __name__ == "__main__":
    titanic_batch_job()