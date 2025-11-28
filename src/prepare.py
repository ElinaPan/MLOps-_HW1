
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    test_size = params["prepare"]["test_size"]
    random_state = params["base"]["random_state"]

    df = pd.read_csv("data/raw/data.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train
    test_df = X_test.copy()
    test_df["target"] = y_test

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    main()
