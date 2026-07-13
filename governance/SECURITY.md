# Segurança & Compliance — Apps de LLM

- **Prompt injection:** guardrails heurísticos (`guardrails/guardrails.py`) cobrem os padrões mais comuns em PT/EN. Em produção, complementar com um classificador dedicado (ex.: Llama Guard, Prompt Guard) para pegar variações que regex não cobre.
- **Extração do system prompt:** o system prompt não deve conter segredos (chaves, credenciais) — trate-o como uma superfície pública, já que técnicas de jailbreak podem eventualmente extraí-lo.
- **PII em conversas:** redação automática antes de logar/persistir (ver `redact_pii`). Avaliar se é necessário criptografar logs de conversa em repouso.
- **Chave de API:** `ANTHROPIC_API_KEY` nunca deve ser commitada — usar variável de ambiente / secret manager. Sem a chave, o sistema usa o provider mock automaticamente (fail-safe, não fail-open para um provider real sem controle).
- **Custo como vetor de ataque:** sem rate limiting, um usuário malicioso pode gerar custo alto de API repetindo perguntas caras. O cache (`cache/semantic_cache.py`) ajuda, mas produção precisa de rate limiting por usuário/IP também.
- **Auditoria:** cada resposta carrega um `trace_id` (`monitoring/tracing.py`) que permite reconstruir exatamente qual contexto foi recuperado e qual prompt foi enviado — necessário para investigar incidentes.
