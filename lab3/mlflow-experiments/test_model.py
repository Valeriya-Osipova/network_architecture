import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

print("=== Тестирование зарегистрированной модели ===")

def test_registered_model():
    # Настраиваем MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Загружаем тестовые данные
    iris = load_iris()
    X_test = iris.data[:10]  # Берем первые 10 samples для теста
    y_true = iris.target[:10]
    
    print("1. Загружаем модель из registry...")
    model_name = "Iris_RandomForest"
    model_version = 1
    
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        print(f"   Модель '{model_name}' версии {model_version} успешно загружена")
    except Exception as e:
        print(f"   ОШИБКА при загрузке модели: {e}")
        return
    
    # Делаем предсказания
    print("2. Делаем предсказания...")
    predictions = model.predict(X_test)
    
    print("\n3. Результаты тестирования:")
    print(f"   Истинные метки:    {y_true}")
    print(f"   Предсказания:      {predictions}")
    print(f"   Точность на тесте: {np.sum(predictions == y_true) / len(y_true):.2f}")
    
    # Логируем тестовый запуск в MLflow
    print("4. Логируем результаты тестирования...")
    with mlflow.start_run(run_name="Model Testing"):
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_metric("test_accuracy", np.sum(predictions == y_true) / len(y_true))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.set_tag("test_type", "deployment_test")
        
        # Логируем пример предсказаний
        for i, (true, pred) in enumerate(zip(y_true, predictions)):
            mlflow.log_metric(f"sample_{i}_true", true)
            mlflow.log_metric(f"sample_{i}_pred", pred)
    
    print("=== Тестирование завершено! ===")

if __name__ == "__main__":
    test_registered_model()
