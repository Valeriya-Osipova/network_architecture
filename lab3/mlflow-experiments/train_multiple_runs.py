import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

print("=== Сравнение нескольких моделей ===")

def train_multiple_models():
    # Загружаем данные
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Iris Model Comparison")
    
    # Пробуем разные модели и параметры
    models_config = [
        {
            "name": "RandomForest_100_5",
            "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "model_type": "RandomForest"}
        },
        {
            "name": "RandomForest_50_3", 
            "model": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
            "params": {"n_estimators": 50, "max_depth": 3, "model_type": "RandomForest"}
        },
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(random_state=42, max_iter=200),
            "params": {"model_type": "LogisticRegression", "C": 1.0, "max_iter": 200}
        },
        {
            "name": "SVM_RBF",
            "model": SVC(random_state=42, kernel='rbf'),
            "params": {"model_type": "SVM", "kernel": "rbf", "C": 1.0}
        }
    ]
    
    results = []
    
    for config in models_config:
        print(f"\nОбучаем модель: {config['name']}")
        
        with mlflow.start_run(run_name=config["name"]):
            # Обучаем модель
            model = config["model"]
            model.fit(X_train, y_train)
            
            # Предсказания и метрики
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Для многоклассовой классификации используем macro averaging
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            
            # Логируем параметры
            for param_name, param_value in config["params"].items():
                mlflow.log_param(param_name, param_value)
            
            # Логируем метрики
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Логируем модель
            mlflow.sklearn.log_model(model, "model")
            
            results.append({
                "model": config["name"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            })
            
            print(f"   Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Выводим сравнение результатов
    print("\n=== Сравнение моделей ===")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{result['model']}: Accuracy={result['accuracy']:.4f}")

if __name__ == "__main__":
    train_multiple_models()
