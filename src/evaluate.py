import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from pathlib import Path
import mlflow


# Add artifact_base_name as a parameter
def evaluate_model(model, X, y, artifact_base_name, reports_dir):
    print("Evaluating model...")
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=["Died", "Survived"], cmap=plt.cm.Blues, ax=ax
    )
    # Update the title to show the specific model
    plt.title(f"{artifact_base_name.split('_')[0]} Confusion Matrix")

    plt.figtext(
        0.5,
        0.02,
        f"Overall Accuracy: {accuracy:.2%}",
        ha="center",
        fontsize=12,
        weight="bold",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5, "edgecolor": "gray"},
    )
    plt.subplots_adjust(bottom=0.15)

    # Construct the dynamic filename
    report_filename = f"{artifact_base_name}_confusion_matrix.png"
    report_path = Path(reports_dir) / report_filename

    plt.savefig(report_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(str(report_path))
    print(f"Confusion matrix saved to: {report_path}")
    plt.close(fig)

    return str(report_path)
