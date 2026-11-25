import re

def chunk_text(text, chunk_size=700, overlap=100):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a sentence/word boundary without making tiny chunks.
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size * 0.6:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def make_chunks(parsed_pages, source, chunk_size=700, overlap=100):
    output = []
    for item in parsed_pages:
        for index, content in enumerate(chunk_text(item["text"], chunk_size, overlap)):
            output.append({
                "text": content,
                "source": source,
                "page": item.get("page"),
                "section": item.get("section"),
                "chunk_index": index,
            })
    return output
