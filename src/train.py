from sklearn.pipeline import Pipeline
from saving_loading.save_model import save_pipeline
def train(model, preprocessor, X, y):
    
    print("Assembling pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("Training model...")
    pipeline.fit(X, y)
    
    # Call the external save process
    print("Saving artifact...")
    save_pipeline(pipeline)
    
    # Return the fitted pipeline to main.py for immediate evaluation
    return pipeline
    