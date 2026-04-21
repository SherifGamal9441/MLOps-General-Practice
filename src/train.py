from src.saving_loading.save_model import save_model


def train(model, X, y, artifact_base_name):
    print("Fitting model...")
    model.fit(X, y)

    print("Saving model artifact...")
    # Add the extension here
    full_filename = f"{artifact_base_name}.joblib"
    save_model(model, filename=full_filename)

    return model
