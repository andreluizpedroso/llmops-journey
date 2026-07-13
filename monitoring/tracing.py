"""
Etapa 17 — Observabilidade & Tracing
------------------------------------------
Gera um "trace" estruturado de cada request: cada etapa da chain
(guardrails -> cache -> retrieval -> geração) vira um span com duração,
metadados e resultado. Em produção isso seria exportado para
LangSmith/Prometheus/Datadog — aqui é logado como JSON estruturado,
que é o formato que qualquer um desses backends consome.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round((self.end_ms or time.time() * 1000) - self.start_ms, 2)


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **metadata):
        s = Span(name=name, start_ms=time.time() * 1000, metadata=metadata)
        try:
            yield s
        finally:
            s.end_ms = time.time() * 1000
            self.spans.append(s)

    def to_json(self) -> str:
        return json.dumps(
            {
                "trace_id": self.trace_id,
                "total_duration_ms": round(sum(s.duration_ms for s in self.spans), 2),
                "spans": [
                    {"name": s.name, "duration_ms": s.duration_ms, "metadata": s.metadata}
                    for s in self.spans
                ],
            },
            ensure_ascii=False,
        )
