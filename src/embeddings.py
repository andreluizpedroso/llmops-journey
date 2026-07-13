"""
Etapa 03 (Fundamentos de NLP) + 06 (Embeddings & Busca Vetorial)
-------------------------------------------------------------------
Gera embeddings dos documentos. Por padrão usa TF-IDF (roda offline,
sem GPU, sem chamadas de rede) — o suficiente para demonstrar o
pipeline de ponta a ponta. A interface é a mesma que um embedder real
(ex.: `text-embedding-3-small` da OpenAI, ou um modelo local via
sentence-transformers) usaria: `embed(texts) -> matriz numpy`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# sklearn não inclui stopwords em português nativamente; sem removê-las,
# palavras de alta frequência (artigos, preposições) dominam a similaridade
# de cosseno e o retriever passa a "achar relevante" qualquer pergunta.
PT_STOPWORDS = [
    "a", "o", "os", "as", "de", "da", "do", "das", "dos", "e", "e", "em",
    "um", "uma", "uns", "umas", "para", "por", "com", "sem", "que", "se",
    "no", "na", "nos", "nas", "ao", "aos", "a", "as", "como", "mais",
    "menos", "muito", "pouco", "qual", "quais", "quanto", "quantos",
    "quanta", "quantas", "meu", "minha", "seu", "sua", "tem", "tem",
    "ser", "esta", "estao", "foi", "sao", "ou", "nao", "sim", "ja", "so",
    "tambem", "isso", "essa", "esse", "este", "esta", "pelo", "pela",
]


def tokenize_preview(text: str, max_tokens: int = 12) -> list[str]:
    """Mini-tokenizador ilustrativo (etapa 03: tokens/NLP). Um LLM real
    usa BPE/SentencePiece; aqui é só para visualizar o conceito."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens[:max_tokens]


@dataclass
class Embedder:
    vectorizer: TfidfVectorizer

    @classmethod
    def fit(cls, documents: list[str]) -> "Embedder":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=4096,
            stop_words=PT_STOPWORDS,
        )
        vectorizer.fit(documents)
        return cls(vectorizer=vectorizer)

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray()
