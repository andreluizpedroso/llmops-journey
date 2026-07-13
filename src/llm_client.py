"""
Etapa 01 (Python) + 05 (APIs de LLM)
--------------------------------------
Abstração de provider de LLM. Suporta a API real da Anthropic quando
`ANTHROPIC_API_KEY` está definida no ambiente, e cai automaticamente
para um provider local determinístico ("mock") quando não está —
assim o projeto inteiro roda offline, sem custo e sem chave, mas o
código de produção é o mesmo que seria usado com o provider real.

Trocar de provider (OpenAI, um modelo open-source via Ollama, etc.)
significa apenas implementar outra classe com a mesma interface.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse: ...


def _estimate_tokens(text: str) -> int:
    """Estimativa grosseira (≈4 chars/token) — suficiente para o cache
    e para custo estimado; em produção usar o tokenizer real do modelo."""
    return max(1, len(text) // 4)


class AnthropicClient(LLMClient):
    """Provider real, usado quando há uma chave de API configurada."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # import tardio: só é exigido quando este provider é usado

        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = (time.perf_counter() - start) * 1000
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            provider="anthropic",
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency, 1),
        )


class MockLLMClient(LLMClient):
    """Provider local, determinístico e gratuito. Extrai a resposta
    diretamente do contexto recuperado pelo RAG, simulando (de forma
    simplificada) o que um LLM real faria com o mesmo prompt — permite
    testar todo o pipeline (retrieval -> prompt -> geração -> guardrails
    -> cache -> observabilidade) sem depender de rede ou custo."""

    def __init__(self, model: str = "mock-local-v1"):
        self.model = model

    def generate(self, system: str, user: str, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()

        context_match = re.search(r"<context>(.*?)</context>", user, re.S)
        question_match = re.search(r"<question>(.*?)</question>", user, re.S)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else user

        if not context.strip():
            text = (
                "Não encontrei essa informação na base de conhecimento "
                "disponível. Recomendo abrir um ticket com o suporte."
            )
        else:
            # heurística simples: pega a(s) sentença(s) do contexto mais
            # relevante(s) para a pergunta (overlap de palavras)
            sentences = re.split(r"(?<=[.!?])\s+", context)
            q_words = set(re.findall(r"\w+", question.lower()))
            scored = sorted(
                sentences,
                key=lambda s: len(q_words & set(re.findall(r"\w+", s.lower()))),
                reverse=True,
            )
            best = [s for s in scored[:2] if s.strip()]
            text = " ".join(best) if best else sentences[0]

        latency = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=text.strip(),
            provider="mock",
            model=self.model,
            input_tokens=_estimate_tokens(system + user),
            output_tokens=_estimate_tokens(text),
            latency_ms=round(latency, 2),
        )


def get_llm_client() -> LLMClient:
    """Factory: escolhe o provider real se houver chave de API; caso
    contrário usa o mock local. Este é o único ponto do projeto que
    decide qual provider está ativo."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    return MockLLMClient()
