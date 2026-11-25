from qdrant_client import QdrantClient, models
from uuid import uuid5, NAMESPACE_URL
from .config import QDRANT_PATH, COLLECTION_NAME

_client = None

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client

def stable_id(payload):
    key = "|".join([
        str(payload.get("source", "")),
        str(payload.get("page", "")),
        str(payload.get("section", "")),
        str(payload.get("chunk_index", "")),
        payload.get("text", "")
    ])
    return str(uuid5(NAMESPACE_URL, key))

def ensure_collection(vector_size):
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            )
        )

def store_chunks(chunks, vectors):
    if not chunks:
        return 0
    ensure_collection(len(vectors[0]))
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(models.PointStruct(
            id=stable_id(chunk),
            vector=vector,
            payload=chunk
        ))
    get_client().upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)

def search_chunks(vector, limit=4):
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        with_payload=True
    )
    return [
        {"text": p.payload["text"], "score": float(p.score),
         "source": p.payload.get("source"), "page": p.payload.get("page"),
         "section": p.payload.get("section"), "chunk_index": p.payload.get("chunk_index")}
        for p in result.points
    ]
