"""
Etapa 13 — Design de APIs (FastAPI)
----------------------------------------
Expõe o assistente de suporte como serviço HTTP, orquestrando todas
as camadas construídas: guardrails de entrada -> cache -> RAG chain ->
tracing -> resposta. Esta é a "porta de entrada" de produção do projeto.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "guardrails", "cache", "monitoring"):
    sys.path.append(str(ROOT / sub))

from semantic_cache import SemanticCache
from tracing import Trace

from guardrails import check_output_safety, run_input_guardrails
from indexer import build_index
from rag_pipeline import RAGChain

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store, embedder = build_index()
    _state["chain"] = RAGChain(store, embedder)
    # threshold calibrado para embeddings TF-IDF: como o TF-IDF é léxico
    # (não semântico), só captura paráfrases que compartilham palavras.
    # Com um embedder denso real (OpenAI/Cohere/sentence-transformers),
    # paráfrases sem overlap léxico ("prazo" ~ "quantos dias") também
    # seriam pegas, e o threshold ficaria mais próximo de 0.85-0.92.
    _state["cache"] = SemanticCache(similarity_threshold=0.35)
    _state["request_count"] = 0
    yield


app = FastAPI(
    title="Aurora Support Assistant API",
    description="Assistente de suporte com RAG — projeto LLMOps Journey",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    cache_hit: str
    latency_ms: float
    trace_id: str


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": "chain" in _state}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    trace = Trace()
    chain: RAGChain = _state["chain"]
    cache: SemanticCache = _state["cache"]

    with trace.span("input_guardrails"):
        guard_result = run_input_guardrails(payload.question)
    if not guard_result.allowed:
        raise HTTPException(status_code=400, detail=f"Requisição bloqueada: {guard_result.reason}")

    question = guard_result.redacted_text or payload.question

    with trace.span("cache_lookup"):
        cached_answer, cache_hit = cache.get(question, chain.embedder.embed)

    if cached_answer is not None:
        answer, sources = cached_answer, []
    else:
        with trace.span("rag_chain", prompt_version=chain.prompt_version):
            result = chain.run(question)
        answer = result.answer
        sources = sorted({s.document.metadata["source"] for s in result.sources})

        with trace.span("output_guardrails"):
            output_check = check_output_safety(answer)
        if not output_check.allowed:
            answer = "Não posso fornecer essa informação."

        cache.set(question, answer, chain.embedder.embed)

    _state["request_count"] += 1

    return ChatResponse(
        answer=answer,
        sources=sources,
        cache_hit=cache_hit,
        latency_ms=round(sum(s.duration_ms for s in trace.spans), 2),
        trace_id=trace.trace_id,
    )


@app.get("/metrics")
def metrics():
    cache: SemanticCache = _state.get("cache")
    return {
        "request_count": _state.get("request_count", 0),
        "cache_stats": cache.stats() if cache else None,
    }
