"""
Etapa 16 — Cache & Otimização de Custo
--------------------------------------------
Duas camadas de cache:
  1. Exato: hash da pergunta normalizada -> resposta (rápido, sem custo
     de embedding, pega perguntas repetidas ao pé da letra).
  2. Semântico: se não houver hit exato, compara o embedding da nova
     pergunta com os embeddings das perguntas já cacheadas — pega
     paráfrases ("quantos dias para reembolso" ~ "prazo de reembolso").

Isso é o que separa um app de LLM caro de um barato em produção: cada
cache hit é uma chamada ao provider que você não paga.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _hash_key(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode()).hexdigest()


@dataclass
class CacheEntry:
    question: str
    answer: str
    vector: np.ndarray
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass
class SemanticCache:
    similarity_threshold: float = 0.92
    ttl_seconds: float = 3600.0
    _exact: dict = field(default_factory=dict)
    _entries: list = field(default_factory=list)

    def get(self, question: str, embed_fn) -> tuple[str | None, str]:
        """Retorna (resposta_ou_None, tipo_de_hit: 'exact'|'semantic'|'miss')."""
        key = _hash_key(question)
        now = time.time()

        entry = self._exact.get(key)
        if entry and (now - entry.created_at) < self.ttl_seconds:
            entry.hits += 1
            return entry.answer, "exact"

        if self._entries:
            query_vec = embed_fn([question])[0]
            best_score, best_entry = -1.0, None
            for e in self._entries:
                if (now - e.created_at) >= self.ttl_seconds:
                    continue
                sim = float(
                    np.dot(query_vec, e.vector)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(e.vector) + 1e-10)
                )
                if sim > best_score:
                    best_score, best_entry = sim, e
            if best_entry and best_score >= self.similarity_threshold:
                best_entry.hits += 1
                return best_entry.answer, "semantic"

        return None, "miss"

    def set(self, question: str, answer: str, embed_fn) -> None:
        key = _hash_key(question)
        vector = embed_fn([question])[0]
        entry = CacheEntry(question=question, answer=answer, vector=vector)
        self._exact[key] = entry
        self._entries.append(entry)

    def stats(self) -> dict:
        total_hits = sum(e.hits for e in self._entries)
        return {"entries": len(self._entries), "total_hits": total_hits}
