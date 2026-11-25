from time import perf_counter
from .document_loader import load_documents, parse_document
from .chunking import make_chunks
from .embeddings import embed_texts
from .vector_store import store_chunks
from .config import DATA_DIR

def index_documents(folder, max_documents=5):
    total_start = perf_counter()
    docs = load_documents(folder, max_documents=max_documents)
    all_chunks = []
    parsed_count = 0
    for doc in docs:
        parsed = parse_document(doc)
        parsed_count += len(parsed)
        all_chunks.extend(make_chunks(parsed, doc["name"]))
    print(f"[PARSE] {len(docs)} docs -> {parsed_count} pages/sections")
    print(f"[CHUNK] {len(all_chunks)} chunks created")
    t = perf_counter()
    vectors = embed_texts([c["text"] for c in all_chunks])
    print(f"[EMBED] {len(vectors)} embeddings in {perf_counter()-t:.2f}s")
    t = perf_counter()
    stored = store_chunks(all_chunks, vectors)
    print(f"[STORE] upserted {stored} points in {perf_counter()-t:.2f}s")
    print(f"[DONE] total {perf_counter()-total_start:.2f}s")
    return {"documents": len(docs), "pages_or_sections": parsed_count,
            "chunks": len(all_chunks), "embeddings": len(vectors), "stored": stored}

if __name__ == "__main__":
    index_documents(DATA_DIR, max_documents=5)
