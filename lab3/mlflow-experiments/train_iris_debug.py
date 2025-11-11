import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import requests
import sys

print("=== Начало отладки ===")

# Сначала проверим доступность сервера
try:
    print("1. Проверяем доступность MLflow сервера...")
    response = requests.get("http://127.0.0.1:5000", timeout=5)
    print(f"   Сервер доступен, статус: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ОШИБКА: Не удалось подключиться к MLflow серверу")
    print("   Убедитесь, что MLflow запущен на порту 5000")
    sys.exit(1)
except Exception as e:
    print(f"   ОШИБКА: {e}")
    sys.exit(1)

def train_iris_model():
    print("2. Загружаем данные Iris...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print("3. Разделяем данные на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("4. Настраиваем MLflow tracking URI...")
    # Используем HTTP вместо HTTPS для отладки
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    print("5. Устанавливаем эксперимент...")
    mlflow.set_experiment("Iris Classification")
    
    print("6. Начинаем запуск MLflow...")
    with mlflow.start_run():
        print("7. Создаем и обучаем модель...")
        n_estimators = 100
        max_depth = 5

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        print("8. Делаем предсказания...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("9. Логируем параметры и метрики...")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", "iris")

        mlflow.log_metric("accuracy", accuracy)

        print("10. Логируем модель...")
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("problem_type", "classification")

        print(f"Модель обучена с точностью: {accuracy:.4f}")

        print("11. Регистрируем модель...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="Iris_RandomForest"
        )
        
        print("=== Обучение успешно завершено! ===")

if __name__ == "__main__":
    train_iris_model()
