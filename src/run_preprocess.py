import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path

from src.preprocessor import preprocess
from src.saving_loading.save_preprocessing import save_data, save_preprocessor

@hydra.main(version_base="1.3", config_path="../config", config_name="model")
def main(cfg: DictConfig):
    print("=== Starting Preprocessing Stage ===")
    
    # 1. Load Data using Hydra Paths
    print("Loading raw data...")
    raw_df = pd.read_csv(cfg.paths.raw_data)
    X = raw_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = raw_df['Survived']

    # 2. Fit and Transform
    preprocessor = preprocess()
    print("Fitting preprocessor and transforming data...")
    X_processed = preprocessor.fit_transform(X)
    X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())


    # 3. Save Everything using Hydra Paths
    preprocessor_file = Path(cfg.paths.preprocessors_dir) / "preprocessor.joblib"
    
    save_data(X_processed_df, y, cfg.paths.processed_X, cfg.paths.processed_y)
    save_preprocessor(preprocessor, preprocessor_file)
    
    print("=== Preprocessing Complete ===")

if __name__ == "__main__":
    main()