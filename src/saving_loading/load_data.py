import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
    y = df['Survived']
    return X, y