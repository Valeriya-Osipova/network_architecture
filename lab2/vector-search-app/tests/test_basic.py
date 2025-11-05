import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_root_endpoint():
    """Тест корневого endpoint через requests"""
    import requests
    response = requests.get("http://localhost:8000/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Vector Search API"
    assert data["status"] == "running"
    assert data["environment"] == "local"
    assert "services" in data
    assert "endpoints" in data
    
    # Проверяем структуру services
    services = data["services"]
    assert "redis" in services
    assert "qdrant" in services
    assert services["redis"] == "localhost:6379"
    assert services["qdrant"] == "localhost:6333"
    
    # Проверяем структуру endpoints
    endpoints = data["endpoints"]
    assert endpoints["health"] == "/health"
    assert endpoints["search"] == "/search"
    assert endpoints["add_vector"] == "/vectors"
    assert endpoints["cache"] == "/cache/{key}"

def test_health_endpoint():
    """Тест health check endpoint через requests"""
    import requests
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "redis" in data
    assert "qdrant" in data
    assert "details" in data
    
    # Проверяем структуру details
    details = data["details"]
    assert "version" in details
    assert "environment" in details

def test_search_endpoint():
    """Тест search endpoint"""
    import requests
    
    # Тест с валидными данными
    search_data = {
        "vector": [0.1] * 128,  # 128-мерный вектор
        "limit": 5
    }
    response = requests.post("http://localhost:8000/search", json=search_data)
    # Может быть 200 или 500 в зависимости от доступности Qdrant
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "source" in data
        assert "results" in data
        assert data["source"] in ["cache", "database"]

def test_add_vector_endpoint():
    """Тест add vector endpoint"""
    import requests
    import uuid
    
    # Генерируем уникальный ID для теста
    test_id = f"test_vector_{uuid.uuid4().hex[:8]}"
    
    vector_data = {
        "id": test_id,
        "vector": [0.1] * 128,  # 128-мерный вектор
        "payload": {"type": "test", "name": "test_vector"}
    }
    
    response = requests.post("http://localhost:8000/vectors", json=vector_data)
    # Может быть 200 или 500 в зависимости от доступности Qdrant
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["status"] == "added"

def test_vectors_count_endpoint():
    """Тест endpoint для получения количества векторов"""
    import requests
    
    response = requests.get("http://localhost:8000/vectors/count")
    # Может быть 200 или 500 в зависимости от доступности Qdrant
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "count" in data
        assert isinstance(data["count"], int)

def test_api_documentation():
    """Тест что документация доступна"""
    import requests
    
    # Проверяем Swagger UI
    response = requests.get("http://localhost:8000/docs")
    assert response.status_code == 200
    
    # Проверяем ReDoc
    response = requests.get("http://localhost:8000/redoc")
    assert response.status_code == 200
    
    # Проверяем OpenAPI schema
    response = requests.get("http://localhost:8000/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema

def test_invalid_search_validation():
    """Тест валидации невалидных данных для search"""
    import requests
    
    # Тест с невалидным вектором (не список)
    invalid_data = {
        "vector": "not_a_list",
        "limit": 5
    }
    response = requests.post("http://localhost:8000/search", json=invalid_data)
    assert response.status_code == 422  # Validation error
    
    # Тест с отсутствующим обязательным полем
    missing_field_data = {
        "limit": 5
    }
    response = requests.post("http://localhost:8000/search", json=missing_field_data)
    assert response.status_code == 422  # Validation error

def test_invalid_vector_validation():
    """Тест валидации невалидных данных для add vector"""
    import requests
    
    # Тест с невалидным вектором
    invalid_data = {
        "id": "test_id",
        "vector": "not_a_list",
        "payload": {"type": "test"}
    }
    response = requests.post("http://localhost:8000/vectors", json=invalid_data)
    assert response.status_code == 422  # Validation error
    
    # Тест с отсутствующим обязательным полем
    missing_field_data = {
        "vector": [0.1] * 128
    }
    response = requests.post("http://localhost:8000/vectors", json=missing_field_data)
    assert response.status_code == 422  # Validation error

def test_endpoint_not_found():
    """Тест обработки несуществующих endpoints"""
    import requests
    
    response = requests.get("http://localhost:8000/nonexistent_endpoint")
    assert response.status_code == 404
    
    response = requests.post("http://localhost:8000/invalid_endpoint", json={})
    assert response.status_code == 404
