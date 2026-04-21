from src.saving_loading.save_model import save_model
def train(model, X, y):
    
    print("Training model...")
    model.fit(X, y)
    
    # Call the external save process
    print("Saving artifact...")
    save_model(model)
    
    # Return the fitted pipeline to main.py for immediate evaluation
    return model
    