import os

class Config:
    """Конфигурация приложения для разных сред"""
    
    # Режим работы: "local" или "kubernetes"
    ENV = os.getenv("APP_ENV", "local")
    
    # Настройки Redis
    if ENV == "kubernetes":
        REDIS_HOST = "redis-master"
    else:
        REDIS_HOST = "localhost"  # Для локальной разработки
    
    REDIS_PORT = 6379
    
    # Настройки Qdrant
    if ENV == "kubernetes":
        QDRANT_HOST = "qdrant"
    else:
        QDRANT_HOST = "localhost"  # Для локальной разработки
    
    QDRANT_PORT = 6333
