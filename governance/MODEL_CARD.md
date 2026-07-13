# Model Card — Aurora Support Assistant (RAG)

## Etapa 20 — Governança, Compliance & System Design

### Visão geral
- **Objetivo:** responder perguntas de suporte sobre políticas da Aurora Cloud (reembolso, limites de uso, segurança, SLA), usando apenas a base de conhecimento oficial — nunca conhecimento genérico do modelo.
- **Arquitetura:** RAG (retrieval-augmented generation), não fine-tuning — a base de conhecimento muda com frequência, então buscar o contexto certo em tempo real é mais seguro do que "gravar" políticas no modelo.
- **Provider:** Anthropic Claude (produção) / mock local determinístico (desenvolvimento e testes, sem custo).

### Dados
- Base de conhecimento: 4 documentos internos de política (`data/knowledge_base/`), sem PII.
- Nenhum dado de conversa de cliente é usado para treinar ou ajustar o modelo.

### Avaliação (ver `eval/evaluate.py`)
| Métrica | Valor de referência | Gate no CI |
|---|---|---|
| Answer coverage | ~86% | mínimo 80% |
| Hallucination rate | 0% | máximo 10% |

**Hallucination rate** aqui mede especificamente perguntas fora da base de
conhecimento que o sistema respondeu como se soubesse — a métrica mais
crítica para um assistente que não pode inventar política de reembolso.

### Limitações conhecidas
- O retriever de demonstração usa TF-IDF (léxico, não semântico) — paráfrases sem overlap de palavras podem não recuperar o contexto certo. Ver `vectordb/README.md` para o caminho de migração para embeddings densos.
- O sistema não deve ser usado para decisões financeiras vinculantes (ex.: aprovar um reembolso automaticamente) — ele informa a política, mas a execução da ação passa por um humano ou por um sistema transacional separado.
- Respostas do provider mock são heurísticas simples, não geração real — usadas apenas para desenvolvimento/CI sem custo.

### Segurança (ver `guardrails/guardrails.py`)
- Detecção de prompt injection (tentativas de sobrescrever o system prompt).
- Redação automática de PII (CPF, e-mail, cartão) antes de logar ou encaminhar a pergunta.
- Filtro de conteúdo na saída do modelo.
- Nenhuma pergunta de usuário é usada para re-treinar o modelo do provider (verificar contrato/DPA do provider usado em produção).

### Privacidade & Compliance
- Conversas podem conter dados pessoais do cliente (mesmo após redação, o contexto pode remeter à identidade) — reter logs pelo prazo mínimo necessário e seguir a política de retenção de dados da LGPD/GDPR.
- Direito ao esquecimento: se um cliente pedir exclusão de dados, isso deve incluir logs de conversas com o assistente, não só a conta.

### Custo & Observabilidade
- Cache exato + semântico (`cache/semantic_cache.py`) reduz chamadas repetidas ao provider.
- Tracing estruturado (`monitoring/tracing.py`) permite auditar exatamente qual contexto foi usado para gerar cada resposta — essencial para investigar reclamações de "o assistente disse algo errado".
