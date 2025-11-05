from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class VectorSearchRequest(BaseModel):
    vector: List[float]
    limit: int = 10

class SearchResult(BaseModel):
    id: str
    score: float
    payload: Optional[Dict[str, Any]] = None

class CacheItem(BaseModel):
    key: str
    value: str
    ttl: Optional[int] = 300

class HealthResponse(BaseModel):
    status: str
    redis: str
    qdrant: str
    details: Optional[Dict[str, Any]] = None

class VectorItem(BaseModel):
    id: str
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None
