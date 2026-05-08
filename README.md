# MLOps-General-Practice

## how to run
1- uv run dvc repro -> to run the preprocessing, train and evaluate
2- uv run python inference.py -> to test the model you trained, pulled from mlflow
3- uv run python src/serve.py -> to start the end point of post prediction, then use bruno to request like:
POST: http://localhost:8000/predict
then in body type the passenger in json format