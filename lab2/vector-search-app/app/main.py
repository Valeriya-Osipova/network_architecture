from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from redis import Redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid
import logging
import os

from .models import VectorSearchRequest, SearchResult, CacheItem, HealthResponse, VectorItem
from .config import Config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Функция инициализации при старте
async def startup():
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    pass

# Инициализация клиентов с учетом конфигурации
redis_client = Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True)
qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)

app = FastAPI(
    title="Vector Search API",
    description="FastAPI приложение для векторного поиска с Redis кешированием",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с веб-интерфейсом"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vector Search API</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .endpoint-card { margin-bottom: 1rem; }
            .response-area { background: #f8f9fa; padding: 1rem; border-radius: 0.375rem; font-family: monospace; }
            .btn-group { margin-bottom: 1rem; }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <div class="row">
                <div class="col-12">
                    <h1 class="mb-4">🔍 Vector Search API</h1>
                    
                    <div class="alert alert-info">
                        <strong>Environment:</strong> <span id="env-status">Loading...</span> |
                        <strong>Redis:</strong> <span id="redis-status">Loading...</span> |
                        <strong>Qdrant:</strong> <span id="qdrant-status">Loading...</span>
                    </div>

                    <div class="btn-group">
                        <a href="/docs" class="btn btn-outline-primary">Swagger Documentation</a>
                        <a href="/redoc" class="btn btn-outline-secondary">ReDoc Documentation</a>
                        <button class="btn btn-outline-success" onclick="testHealth()">Test Health</button>
                    </div>

                    <div class="row">
                        <!-- Health Check -->
                        <div class="col-md-6">
                            <div class="card endpoint-card">
                                <div class="card-header">
                                    <h5 class="card-title mb-0">🏥 Health Check</h5>
                                </div>
                                <div class="card-body">
                                    <button class="btn btn-primary btn-sm" onclick="testHealth()">Test /health</button>
                                    <div class="response-area mt-2" id="health-response" style="min-height: 100px;">
                                        Click button to test
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Cache Operations -->
                        <div class="col-md-6">
                            <div class="card endpoint-card">
                                <div class="card-header">
                                    <h5 class="card-title mb-0">💾 Cache Operations</h5>
                                </div>
                                <div class="card-body">
                                    <div class="mb-2">
                                        <input type="text" class="form-control form-control-sm mb-1" id="cache-key" placeholder="Key" value="test_key">
                                        <input type="text" class="form-control form-control-sm mb-1" id="cache-value" placeholder="Value" value="test_value">
                                        <button class="btn btn-success btn-sm" onclick="setCache()">Set Cache</button>
                                        <button class="btn btn-info btn-sm" onclick="getCache()">Get Cache</button>
                                    </div>
                                    <div class="response-area" id="cache-response" style="min-height: 100px;">
                                        Cache operations will appear here
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Vector Search -->
                        <div class="col-md-6">
                            <div class="card endpoint-card">
                                <div class="card-header">
                                    <h5 class="card-title mb-0">🔍 Vector Search</h5>
                                </div>
                                <div class="card-body">
                                    <button class="btn btn-warning btn-sm" onclick="testSearch()">Test Search</button>
                                    <div class="response-area mt-2" id="search-response" style="min-height: 150px;">
                                        Search results will appear here
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Add Vector -->
                        <div class="col-md-6">
                            <div class="card endpoint-card">
                                <div class="card-header">
                                    <h5 class="card-title mb-0">➕ Add Vector</h5>
                                </div>
                                <div class="card-body">
                                    <button class="btn btn-secondary btn-sm" onclick="addVector()">Add Test Vector</button>
                                    <div class="response-area mt-2" id="vector-response" style="min-height: 100px;">
                                        Vector operations will appear here
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Load initial status
            async function loadStatus() {
                try {
                    const response = await fetch('/');
                    const data = await response.json();
                    document.getElementById('env-status').textContent = data.environment;
                    document.getElementById('redis-status').textContent = data.services.redis;
                    document.getElementById('qdrant-status').textContent = data.services.qdrant;
                } catch (error) {
                    console.error('Error loading status:', error);
                }
            }

            // Test health endpoint
            async function testHealth() {
                const responseArea = document.getElementById('health-response');
                responseArea.textContent = 'Testing...';
                
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    responseArea.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    responseArea.textContent = 'Error: ' + error.message;
                }
            }

            // Set cache
            async function setCache() {
                const key = document.getElementById('cache-key').value;
                const value = document.getElementById('cache-value').value;
                const responseArea = document.getElementById('cache-response');
                responseArea.textContent = 'Setting cache...';
                
                try {
                    const response = await fetch('/cache', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ key, value, ttl: 300 })
                    });
                    const data = await response.json();
                    responseArea.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    responseArea.textContent = 'Error: ' + error.message;
                }
            }

            // Get cache
            async function getCache() {
                const key = document.getElementById('cache-key').value;
                const responseArea = document.getElementById('cache-response');
                responseArea.textContent = 'Getting cache...';
                
                try {
                    const response = await fetch('/cache/' + key);
                    const data = await response.json();
                    responseArea.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    responseArea.textContent = 'Error: ' + error.message;
                }
            }

            // Test vector search
            async function testSearch() {
                const responseArea = document.getElementById('search-response');
                responseArea.textContent = 'Searching...';
                
                try {
                    // Create a simple 128-dimensional vector
                    const vector = Array(128).fill(0).map((_, i) => Math.random() * 0.1);
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ vector, limit: 5 })
                    });
                    const data = await response.json();
                    responseArea.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    responseArea.textContent = 'Error: ' + error.message;
                }
            }

            // Add test vector
            async function addVector() {
                const responseArea = document.getElementById('vector-response');
                responseArea.textContent = 'Adding vector...';
                
                try {
                    const vector = Array(128).fill(0).map((_, i) => Math.random());
                    const response = await fetch('/vectors', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            id: 'test_vector_' + Date.now(),
                            vector: vector,
                            payload: { type: 'test', timestamp: new Date().toISOString() }
                        })
                    });
                    const data = await response.json();
                    responseArea.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    responseArea.textContent = 'Error: ' + error.message;
                }
            }

            // Initialize
            loadStatus();
        </script>
    </body>
    </html>
    """
    return html_content

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
