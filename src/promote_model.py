import os
import mlflow
from mlflow.tracking import MlflowClient
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

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient(tracking_uri=tracking_uri)

    print("Searching for the best model...")
    
    experiment = client.get_experiment_by_name("Titanic_Experiments")
    if not experiment:
        print("Experiment 'Titanic_Cloud_Final' not found!")
        return
    exp_ids = [experiment.experiment_id]

    # Grab the top 10 highest-scoring runs instead of just 1
    runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=["metrics.accuracy DESC"],
        max_results=10
    )
    print(f"Searching in experiment IDs: {exp_ids}")
    for run in runs:
        print(f"  Run {run.info.run_id} | experiment: {run.info.experiment_id} | artifacts: {[a.path for a in client.list_artifacts(run.info.run_id)]}")
    if not runs:
        print("No runs found! Make sure your pipeline ran successfully.")
        return

    # 2. BULLETPROOF CHECK: Verify the artifact actually exists before promoting
    best_run = None
    for run in runs:
        run_id = run.info.run_id
        accuracy = run.data.metrics.get('accuracy', 'N/A')
        
        # Check the actual artifacts attached to this specific run
        def has_model_artifact(client, run_id):
            artifacts = client.list_artifacts(run_id)
            for a in artifacts:
                if a.path == "model" or a.path.startswith("model/"):
                    return True
                if a.is_dir:
                    # check one level deeper
                    sub = client.list_artifacts(run_id, a.path)
                    if any(s.path.startswith("model") for s in sub):
                        return True
            return False
        
        if has_model_artifact(client, run_id):
            print(f"Found valid run with model! ID: {run_id} | Accuracy: {accuracy}")
            best_run = run
            break  # Stop searching, we found our champion
        else:
            print(f"Skipping Run {run_id} (Accuracy: {accuracy}) - No 'model' artifact found.")

    if not best_run:
        print("Searched the top runs, but NONE of them contained a saved model artifact.")
        print("TIP: Check your train.py to ensure mlflow.sklearn.log_model() is executing correctly.")
        return

    best_run_id = best_run.info.run_id

    # 3. Register the model
    model_name = "Titanic_Survival_Model"
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f"Registering model as '{model_name}'...")
    model_version_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # 4. Promote with Aliases
    alias = "Production"
    print(f"Tagging version {model_version_details.version} with alias '@{alias}'...")
    
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(model_version_details.version)
    )
    
    print(f"🎉 Model successfully promoted to '{alias}'!")

if __name__ == "__main__":
    main()