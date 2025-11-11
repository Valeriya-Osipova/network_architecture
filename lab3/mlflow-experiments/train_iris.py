import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Для обхода SSL ошибок (временно)
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

print("start training")

def train_iris_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Используем правильный URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Iris Classification")
    
    with mlflow.start_run():

        n_estimators = 100
        max_depth = 5

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", "iris")

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("problem_type", "classification")

        print(f"Model trained with accuracy: {accuracy:.4f}")

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="Iris_RandomForest"
        )

if __name__ == "__main__":
    train_iris_model()
