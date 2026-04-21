from sklearn.pipeline import Pipeline
def train(model, preprocessor, X, y):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 4. Train and Evaluate
    # The pipeline handles all preprocessing and training automatically
    pipeline.fit(X, y)