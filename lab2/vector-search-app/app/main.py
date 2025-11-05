from fastapi import FastAPI, HTTPException
from redis import Redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid
import logging

from .models import VectorSearchRequest, SearchResult, CacheItem, HealthResponse, VectorItem
from .config import Config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Search API",
    description="FastAPI приложение для векторного поиска с Redis кешированием",
    version="1.0.0"
)

# Инициализация клиентов с учетом конфигурации
redis_client = Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True)
qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)

# Создание коллекции при старте (если не существует)
@app.on_event("startup")
async def startup_event():
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if "documents" not in collection_names:
            qdrant_client.create_collection(
                collection_name="documents",
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            logger.info("Создана новая коллекция Qdrant 'documents'")
        else:
            logger.info("Коллекция Qdrant 'documents' уже существует")
            
    except Exception as e:
        logger.warning(f"Ошибка при инициализации Qdrant: {e}")

@app.get("/")
async def root():
    return {
        "message": "Vector Search API", 
        "status": "running",
        "environment": Config.ENV,
        "services": {
            "redis": f"{Config.REDIS_HOST}:{Config.REDIS_PORT}",
            "qdrant": f"{Config.QDRANT_HOST}:{Config.QDRANT_PORT}"
        },
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "add_vector": "/vectors",
            "cache": "/cache/{key}"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Проверка Redis
        try:
            redis_client.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
        
        # Проверка Qdrant
        try:
            qdrant_client.get_collections()
            qdrant_status = "connected"
        except:
            qdrant_status = "disconnected"
        
        status = "healthy" if redis_status == "connected" and qdrant_status == "connected" else "degraded"
        
        return HealthResponse(
            status=status,
            redis=redis_status,
            qdrant=qdrant_status,
            details={
                "version": "1.0.0",
                "environment": Config.ENV
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unavailable: {str(e)}"
        )

@app.post("/search")
async def search_vectors(request: VectorSearchRequest):
    """Поиск похожих векторов"""
    try:
        # Проверяем кеш
        cache_key = f"search:{hash(tuple(request.vector))}:{request.limit}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            logger.info(f"Результат найден в кеше: {cache_key}")
            return {"source": "cache", "results": eval(cached_result)}
        
        # Выполняем поиск в Qdrant
        search_result = qdrant_client.search(
            collection_name="documents",
            query_vector=request.vector,
            limit=request.limit
        )
        
        results = [
            SearchResult(
                id=hit.id,
                score=hit.score,
                payload=hit.payload
            ).dict() for hit in search_result
        ]
        
        # Сохраняем в кеш на 5 минут
        redis_client.setex(cache_key, 300, str(results))
        logger.info(f"Результат сохранен в кеш: {cache_key}")
        
        return {"source": "database", "results": results}
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/vectors")
async def add_vector(item: VectorItem):
    """Добавление нового вектора"""
    try:
        point = PointStruct(
            id=item.id,
            vector=item.vector,
            payload=item.payload or {}
        )
        
        qdrant_client.upsert(
            collection_name="documents",
            wait=True,
            points=[point]
        )
        
        logger.info(f"Вектор добавлен: {item.id}")
        return {"id": item.id, "status": "added"}
    
    except Exception as e:
        logger.error(f"Failed to add vector: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add vector: {str(e)}")

@app.post("/cache")
async def cache_item(item: CacheItem):
    """Добавление элемента в кеш"""
    try:
        redis_client.setex(item.key, item.ttl, item.value)
        logger.info(f"Элемент закеширован: {item.key}")
        return {"status": "cached", "key": item.key}
    except Exception as e:
        logger.error(f"Cache failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache failed: {str(e)}")

@app.get("/cache/{key}")
async def get_cached_item(key: str):
    """Получение элемента из кеша"""
    try:
        value = redis_client.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"key": key, "value": value}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache retrieval failed: {str(e)}")

@app.get("/vectors/count")
async def get_vectors_count():
    """Получение количества векторов в коллекции"""
    try:
        collection_info = qdrant_client.get_collection("documents")
        return {"count": collection_info.points_count}
    except Exception as e:
        logger.error(f"Failed to get vectors count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get vectors count: {str(e)}")
