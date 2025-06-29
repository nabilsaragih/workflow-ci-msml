import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="personality_preprocessing", help="Path ke folder dataset")
args = parser.parse_args()

base_path = os.path.dirname(__file__)
data_dir = os.path.join(base_path, args.data_path)

X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()

mlflow.sklearn.autolog()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model dilatih dan dicatat dengan autolog.")
