import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from pathlib import Path
import mlflow

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"


# Add artifact_base_name as a parameter
def evaluate_model(model, X, y, artifact_base_name):
    print("Evaluating model...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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
    report_path = REPORTS_DIR / report_filename

    plt.savefig(report_path, bbox_inches="tight", dpi=300)  # save first
    mlflow.log_artifact(report_path)
    print(f"Confusion matrix saved to: {report_path}")
    plt.close(fig)

    return report_path
