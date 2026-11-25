from pathlib import Path
from time import perf_counter
from pypdf import PdfReader
from docx import Document

SUPPORTED = {".pdf", ".docx", ".txt"}

def load_documents(folder, max_documents=5):
    folder = Path(folder)
    documents = []
    start = perf_counter()
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        documents.append({"path": str(path), "name": path.name, "suffix": path.suffix.lower()})
    if max_documents is not None:
        documents = documents[:max_documents]
    print(f"[LOAD] {len(documents)} documents selected in {perf_counter()-start:.2f}s")
    return documents

def parse_document(document):
    path = Path(document["path"])
    pages = []
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": page_no, "section": "PDF page"})
    elif path.suffix.lower() == ".docx":
        doc = Document(str(path))
        current, section = [], "Document"
        for para in doc.paragraphs:
            txt = para.text.strip()
            if not txt:
                continue
            if para.style.name.startswith("Heading"):
                if current:
                    pages.append({"text": "\\n".join(current), "page": None, "section": section})
                    current = []
                section = txt
            else:
                current.append(txt)
        if current:
            pages.append({"text": "\\n".join(current), "page": None, "section": section})
    else:
        text = path.read_text(encoding="utf-8")
        pages.append({"text": text, "page": 1, "section": "Text file"})
    return pages
