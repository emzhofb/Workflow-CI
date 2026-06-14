import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
import os

def load_preprocessed_data(data_dir):
  """Load data yang sudah dipreprocess."""
  train = pd.read_csv(os.path.join(data_dir, "train.csv"))
  test = pd.read_csv(os.path.join(data_dir, "test.csv"))

  X_train = train.drop('target', axis=1)
  y_train = train['target']
  X_test = test.drop('target', axis=1)
  y_test = test['target']

  return X_train, X_test, y_train, y_test

def train_model():
  """Melatih model dengan MLflow autolog."""

  X_train, X_test, y_train, y_test = load_preprocessed_data(
    "../dataset_preprocessing"
  )

  mlflow.set_experiment("MSML-CI-Experiment")
  mlflow.sklearn.autolog()

  with mlflow.start_run(run_name="RandomForest-CI"):
    model = RandomForestClassifier(
      n_estimators=100,
      max_depth=10,
      random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

if __name__ == "__main__":
  train_model()
