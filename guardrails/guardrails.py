"""
Etapa 15 — Guardrails & Segurança
--------------------------------------
Camada de proteção entre o usuário e o LLM:
  - Detecção heurística de prompt injection ("ignore as instruções
    anteriores", tentativas de extrair o system prompt, etc.)
  - Filtro de PII na entrada (CPF, e-mail, cartão) — evita logar dados
    sensíveis sem necessidade e alerta antes de enviar ao provider.
  - Filtro de conteúdo básico na saída do modelo.

Estas são heurísticas — em produção, complementar com um classificador
dedicado (ex.: Llama Guard) para os casos que regex não pega.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore\s+(as|todas as)?\s*instru[çc][õo]es\s+anteriores",
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"revele\s+(o\s+)?system\s*prompt",
    r"reveal\s+(the\s+)?system\s*prompt",
    r"você\s+agora\s+é\s+um",
    r"disregard\s+(all\s+)?(your\s+)?(rules|guidelines)",
    r"modo\s+desenvolvedor|developer\s+mode",
    r"finja\s+que\s+(você\s+)?não\s+tem\s+regras",
]

CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

BLOCKED_OUTPUT_TERMS = ["como fabricar explosivos", "como hackear"]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    redacted_text: str | None = None
    pii_found: list[str] | None = None


def check_prompt_injection(user_input: str) -> GuardrailResult:
    lowered = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(allowed=False, reason=f"possível prompt injection ({pattern})")
    return GuardrailResult(allowed=True)


def redact_pii(text: str) -> GuardrailResult:
    found = []
    redacted = text

    if CPF_PATTERN.search(redacted):
        found.append("CPF")
        redacted = CPF_PATTERN.sub("[CPF REDIGIDO]", redacted)
    if EMAIL_PATTERN.search(redacted):
        found.append("e-mail")
        redacted = EMAIL_PATTERN.sub("[E-MAIL REDIGIDO]", redacted)
    if CREDIT_CARD_PATTERN.search(redacted):
        found.append("cartão")
        redacted = CREDIT_CARD_PATTERN.sub("[CARTÃO REDIGIDO]", redacted)

    return GuardrailResult(allowed=True, redacted_text=redacted, pii_found=found or None)


def check_output_safety(model_output: str) -> GuardrailResult:
    lowered = model_output.lower()
    for term in BLOCKED_OUTPUT_TERMS:
        if term in lowered:
            return GuardrailResult(allowed=False, reason=f"conteúdo bloqueado ({term})")
    return GuardrailResult(allowed=True)


def run_input_guardrails(user_input: str) -> GuardrailResult:
    """Pipeline de guardrails de entrada: injection primeiro (bloqueia),
    depois PII (redige mas não bloqueia)."""
    injection = check_prompt_injection(user_input)
    if not injection.allowed:
        return injection

    pii = redact_pii(user_input)
    return pii
