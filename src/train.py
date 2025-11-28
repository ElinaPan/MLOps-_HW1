
import pandas as pd
import yaml
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

import mlflow
import mlflow.sklearn

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()

    model_type = params["train"]["model_type"]
    C = params["train"]["C"]
    max_iter = params["train"]["max_iter"]
    random_state = params["base"]["random_state"]

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)

    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", acc)

        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")

        mlflow.sklearn.log_model(model, "model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
