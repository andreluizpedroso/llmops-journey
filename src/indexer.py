"""
Etapa 07 — RAG (parte de indexação)
---------------------------------------
Lê os documentos da base de conhecimento, faz chunking e indexa no
vector store. Roda uma vez (ou a cada atualização da base) — não a
cada request.
"""
from __future__ import annotations

from pathlib import Path

from embeddings import Embedder
from vector_store import Document, InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Chunking simples por caracteres com overlap, para não cortar
    frases importantes no meio entre dois chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_and_chunk_kb() -> list[Document]:
    documents = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            documents.append(
                Document(
                    doc_id=f"{path.stem}::chunk{i}",
                    text=chunk,
                    metadata={"source": path.name, "chunk_index": i},
                )
            )
    return documents


def build_index() -> tuple[InMemoryVectorStore, Embedder]:
    documents = load_and_chunk_kb()
    embedder = Embedder.fit([d.text for d in documents])
    vectors = embedder.embed([d.text for d in documents])

    store = InMemoryVectorStore()
    store.add(documents, vectors)
    return store, embedder


if __name__ == "__main__":
    store, embedder = build_index()
    print(f"Indexados {len(store.documents)} chunks de {len(list(KB_DIR.glob('*.md')))} documentos.")
    for doc in store.documents[:3]:
        print(f"- [{doc.metadata['source']}] {doc.text[:80]}...")
