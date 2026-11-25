from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts):
    if not texts:
        return []
    vectors = get_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False
    )
    return vectors.tolist()

def embed_query(question):
    return embed_texts([question])[0]
