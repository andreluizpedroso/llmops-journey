"""
Etapa 06 — Embeddings & Busca Vetorial
------------------------------------------
Vector store minimalista em memória (busca por cosseno em numpy).
Em produção isso seria substituído por Pinecone/Weaviate/Milvus — ver
`vectordb/README.md` para como essa troca seria feita sem mudar o
resto do pipeline (a interface `search(query_vector, k)` é a mesma).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict


@dataclass
class SearchResult:
    document: Document
    score: float


@dataclass
class InMemoryVectorStore:
    documents: list[Document] = field(default_factory=list)
    vectors: np.ndarray | None = None

    def add(self, documents: list[Document], vectors: np.ndarray) -> None:
        self.documents.extend(documents)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        if self.vectors is None or len(self.documents) == 0:
            return []

        # similaridade de cosseno vetorizada
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        doc_norms = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        scores = doc_norms @ query_norm

        top_k_idx = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(document=self.documents[i], score=float(scores[i]))
            for i in top_k_idx
        ]
