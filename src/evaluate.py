import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from pathlib import Path

# Setup paths dynamically (assuming this file is in 'src')
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"

def evaluate_model(model, X, y, filename="confusion_matrix.png"):
    """
    Evaluates the trained model, generates a confusion matrix with accuracy,
    and saves the plot to the reports directory.
    """
    print("Evaluating model...")
    
    # Ensure the reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate the confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        model, 
        X, 
        y, 
        display_labels=['Died', 'Survived'], 
        cmap=plt.cm.Blues,
        ax=ax
    )
    plt.title("Titanic Survival Confusion Matrix")
    
    # Add accuracy text at the bottom of the figure
    # plt.figtext places text relative to the entire figure (x=0.5 is center, y=0.02 is near the bottom)
    plt.figtext(
        0.5, 0.02, 
        f"Overall Accuracy: {accuracy:.2%}", 
        ha="center", 
        fontsize=12, 
        weight="bold",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5, "edgecolor": "gray"}
    )
    
    # Adjust layout to make room for the text at the bottom
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    report_path = REPORTS_DIR / filename
    plt.savefig(report_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to: {report_path}")
    
    # Close the plot to free up memory
    plt.close(fig)
    
    return report_path