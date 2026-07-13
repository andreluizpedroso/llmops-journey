"""
Etapa 08 — Frameworks (LangChain / LlamaIndex)
---------------------------------------------------
Este projeto implementa o padrão de "chain" manualmente (sem depender
de um framework pesado), para deixar explícito o que o LangChain faz
por baixo: compor etapas (retriever -> prompt -> llm -> parser) em um
pipeline único e reexecutável. Trocar por `langchain.chains.RetrievalQA`
ou um `QueryEngine` do LlamaIndex é uma troca de implementação, não de
arquitetura — a assinatura `RAGChain.run(question)` continua a mesma.
"""
from __future__ import annotations

from dataclasses import dataclass

from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, get_llm_client
from prompts import build_prompt
from vector_store import InMemoryVectorStore, SearchResult


@dataclass
class RAGResult:
    answer: str
    sources: list[SearchResult]
    llm_response: LLMResponse
    prompt_version: str


class RAGChain:
    def __init__(
        self,
        store: InMemoryVectorStore,
        embedder: Embedder,
        llm: LLMClient | None = None,
        prompt_version: str = "v2",
        top_k: int = 3,
        min_score: float = 0.1,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm or get_llm_client()
        self.prompt_version = prompt_version
        self.top_k = top_k
        self.min_score = min_score

    def run(self, question: str) -> RAGResult:
        # 1. Retrieval
        query_vector = self.embedder.embed([question])[0]
        results = self.store.search(query_vector, k=self.top_k)

        # Filtra chunks irrelevantes: sem isso, o retriever sempre devolve
        # os top-k mesmo quando a pergunta não tem nada a ver com a base
        # (ex.: pergunta fora de domínio), o que induz o modelo a "alucinar"
        # uma resposta a partir de contexto sem relação com a pergunta.
        relevant = [r for r in results if r.score >= self.min_score]
        context = "\n\n".join(r.document.text for r in relevant)

        # 2. Prompt
        system, user = build_prompt(context, question, version=self.prompt_version)

        # 3. Geração
        response = self.llm.generate(system, user)

        return RAGResult(
            answer=response.text,
            sources=relevant,
            llm_response=response,
            prompt_version=self.prompt_version,
        )
