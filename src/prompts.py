"""
Etapa 02 — Prompt Engineering
--------------------------------
Prompts são código: versionados, testáveis e documentados. Cada
template tem uma versão explícita para permitir comparação A/B e
rollback (ver eval/evaluate.py, que roda o mesmo dataset contra
diferentes versões de prompt).
"""

RAG_SYSTEM_PROMPT_V1 = """\
Você é o assistente de suporte da Aurora Cloud. Responda SOMENTE com
base no contexto fornecido. Se a resposta não estiver no contexto,
diga que não encontrou a informação e sugira abrir um ticket de suporte.
Nunca invente números, prazos ou políticas que não estejam no contexto.
Seja direto e cite o dado exato (prazo, percentual, limite) quando existir.
"""

RAG_USER_TEMPLATE_V1 = """\
<context>
{context}
</context>

<question>
{question}
</question>

Responda à pergunta acima usando apenas o contexto fornecido.
"""

# v2: versão mais restritiva, adicionada após observar respostas
# levemente evasivas no eval — pede resposta em uma frase quando possível
RAG_SYSTEM_PROMPT_V2 = RAG_SYSTEM_PROMPT_V1 + (
    "\nSempre que possível, responda em uma única frase objetiva, "
    "citando o número/prazo exato do contexto."
)

PROMPT_VERSIONS = {
    "v1": (RAG_SYSTEM_PROMPT_V1, RAG_USER_TEMPLATE_V1),
    "v2": (RAG_SYSTEM_PROMPT_V2, RAG_USER_TEMPLATE_V1),
}


def build_prompt(context: str, question: str, version: str = "v2") -> tuple[str, str]:
    system, user_template = PROMPT_VERSIONS[version]
    user = user_template.format(context=context, question=question)
    return system, user
