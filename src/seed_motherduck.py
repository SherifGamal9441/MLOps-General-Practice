import os
import duckdb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def seed_database():
    print("Connecting to MotherDuck...")
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise ValueError("MOTHERDUCK_TOKEN not found in environment.")
        
    # Connect to the default MotherDuck database
    con = duckdb.connect(f"md:?motherduck_token={token}")

    # Define path to the test dataset
    root_dir = Path(__file__).resolve().parent.parent
    test_data_path = root_dir / "data" / "raw" / "test.csv"

    if not test_data_path.exists():
        raise FileNotFoundError(f"Could not find test data at {test_data_path}")

    print("Loading test data into MotherDuck table 'raw_titanic_test'...")
    # DuckDB can natively read CSVs and create tables in one SQL command
    con.execute(f"CREATE TABLE IF NOT EXISTS raw_titanic_test AS SELECT * FROM read_csv_auto('{test_data_path}')")
    
    count = con.execute("SELECT COUNT(*) FROM raw_titanic_test").fetchone()[0]
    print(f"Successfully loaded {count} rows into MotherDuck.")

if __name__ == "__main__":
    seed_database()