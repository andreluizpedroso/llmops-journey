# LLMOps Journey — Assistente de Suporte com RAG

[![CI/CD](https://github.com/andreluizpedroso/llmops-journey/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/andreluizpedroso/llmops-journey/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Anthropic Claude](https://img.shields.io/badge/LLM-Anthropic%20Claude-D97757)

🇺🇸 [Read in English](README.md)

Projeto de ponta a ponta construído seguindo as **20 etapas** do
roadmap *"A Jornada de LLMOps"*. **Roda 100% offline, sem custo e sem
chave de API** por padrão — usa um provider LLM mock local
determinístico — e troca para o provider real da Anthropic com uma
única variável de ambiente, sem mudar nenhuma linha de código.

**Caso de uso escolhido:** assistente de suporte que responde perguntas
sobre políticas de uma empresa fictícia (Aurora Cloud — reembolso,
limites de uso, segurança, SLA) usando **RAG** (retrieval-augmented
generation). É o padrão mais comum de app de LLM em produção hoje, e
foi escolhido de propósito por expor os problemas reais de LLMOps:
alucinação, custo, latência, segurança — não só "chamar uma API de
LLM".

---

## Índice

1. [Como as peças se conectam](#como-as-peças-se-conectam)
2. [Etapa por etapa: o que foi feito e por quê](#etapa-por-etapa-o-que-foi-feito-e-por-quê)
3. [Como rodar o projeto](#como-rodar-o-projeto)
4. [Um bug real que apareci enquanto construía isso](#um-bug-real-que-apareceu-enquanto-construía-isso)
5. [Decisões de design e trade-offs](#decisões-de-design-e-trade-offs)
6. [Estratégia de testes e avaliação](#estratégia-de-testes-e-avaliação)
7. [O que é real vs. o que é template](#o-que-é-real-vs-o-que-é-template)
8. [Troubleshooting](#troubleshooting)
9. [Caminho para produção real](#caminho-para-produção-real)

---

## Como as peças se conectam

```
data/knowledge_base/*.md          src/prompts.py (v1/v2)
        │  documentos-fonte              │  templates versionados
        ▼                                │
src/indexer.py  ── chunking + indexação  │
        │                                │
        ▼                                │
src/embeddings.py (TF-IDF)               │
        │                                │
        ▼                                │
src/vector_store.py (busca por cosseno)  │
        │                                │
        └──────────► src/rag_pipeline.py ◄┘
                            │
                            ▼
                    src/llm_client.py
                    (Anthropic real OU mock local)
                            │
                            ▼
                      api/main.py (FastAPI)
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
guardrails/            cache/                monitoring/
guardrails.py          semantic_cache.py     tracing.py
(injection + PII)      (exato + semântico)   (spans por request)
        │
        ▼
tests/test_api.py  ──  eval/evaluate.py  ──  .github/workflows/ci-cd.yml
(8 testes)              (coverage/hallucination)  (gates de qualidade)
```

O request de um usuário passa, em ordem, por: **guardrails de entrada**
(bloqueia prompt injection, redige PII) → **cache** (exato, depois
semântico) → se não houver hit, **RAG chain completa** (retrieval →
prompt → geração) → **guardrails de saída** → resposta. Cada etapa
desse caminho vira um *span* de tracing, visível no `trace_id` que
volta na resposta.

---

## Etapa por etapa: o que foi feito e por quê

### Fase 01 — Fundamentos

**01. Python** — toda a lógica do projeto.

**02. Prompt Engineering** (`src/prompts.py`)
Os prompts são tratados como código: versionados (`v1`, `v2`) num
dicionário `PROMPT_VERSIONS`, não como strings soltas espalhadas pelo
projeto. O `v1` é o prompt básico ("responda só com base no contexto,
diga que não sabe se não tiver a informação"); o `v2` adiciona uma
instrução extra pedindo respostas objetivas de uma frase, citando o
número/prazo exato — uma iteração que, em um projeto real, teria vindo
de observar respostas evasivas no `eval/evaluate.py`. `build_prompt()`
é a função que qualquer parte do sistema chama para montar o prompt
final; trocar de versão em produção (para comparação A/B) é passar um
parâmetro, não editar strings em produção.

**03. Fundamentos de NLP** (`src/embeddings.py`)
Tem uma função `tokenize_preview()` só para ilustrar o conceito de
tokenização (um LLM real usa BPE/SentencePiece, não split por espaço).
A parte funcional de verdade é o `Embedder`, baseado em TF-IDF do
scikit-learn — decisão deliberada de rodar offline sem GPU nem chamada
de rede (ver seção de limitações abaixo sobre por que isso é
lexical, não semântico).

**04. Git & Versionamento** (`.gitignore`)
Ignora segredos (`ANTHROPIC_API_KEY`) e artefatos de cache/teste.

**05. APIs de LLM** (`src/llm_client.py`)
Este é um dos módulos centrais do projeto. Define uma interface
abstrata `LLMClient` com um único método `generate(system, user,
max_tokens) -> LLMResponse`, e duas implementações:
- `AnthropicClient`: chama a API real da Anthropic (`claude-sonnet-4-6`),
  usada quando `ANTHROPIC_API_KEY` está definida no ambiente.
- `MockLLMClient`: provider local, determinístico e gratuito. Em vez de
  gerar texto livre, ele extrai do `<context>` a(s) sentença(s) com
  maior sobreposição de palavras com a `<question>` — uma heurística
  simples, mas suficiente para testar **todo** o pipeline (retrieval →
  prompt → geração → guardrails → cache → tracing) sem custo e sem
  depender de rede.

A função `get_llm_client()` é uma factory: decide automaticamente qual
provider usar, checando se a variável de ambiente existe. É o único
ponto de decisão do projeto inteiro — todo o resto do código (RAG
chain, API) não sabe nem precisa saber qual provider está ativo.

### Fase 02 — Construindo com LLMs

**06. Embeddings & Busca Vetorial** (`src/embeddings.py`, `src/vector_store.py`)
`Embedder.fit()` treina um `TfidfVectorizer` (unigramas + bigramas, até
4096 features) sobre os documentos da base de conhecimento.
`InMemoryVectorStore` guarda os vetores em uma matriz numpy e faz busca
por similaridade de cosseno vetorizada (`search(query_vector, k)`).
A interface (`add`, `search`) é desenhada para ser exatamente a mesma
que um vector DB real (Pinecone, Weaviate) implementaria — ver
`vectordb/README.md` para o guia de migração.

**07. RAG** (`src/indexer.py`, `src/rag_pipeline.py`)
`indexer.py` lê os `.md` da base de conhecimento, faz chunking por
caracteres com overlap (400 caracteres, 80 de overlap — evita cortar
uma frase importante bem na fronteira entre dois chunks) e indexa no
vector store. `rag_pipeline.py` implementa a `RAGChain`: recebe uma
pergunta, busca os `top_k=3` chunks mais similares, **filtra por um
limiar mínimo de relevância** (`min_score=0.1`), monta o prompt com o
contexto filtrado, e chama o LLM. O filtro de relevância é o que evita
que o sistema tente responder perguntas fora de domínio — ver a seção
de bug real abaixo, porque essa não é uma decisão teórica, foi um
problema que apareceu e foi corrigido durante a construção deste
projeto.

**08. Frameworks (LangChain / LlamaIndex)** (`src/rag_pipeline.py`)
A `RAGChain` implementa manualmente o padrão de "chain" que frameworks
como LangChain (`RetrievalQA`) ou LlamaIndex (`QueryEngine`) oferecem
prontos. A decisão de não usar um framework pesado aqui foi deliberada:
o projeto fica sem uma dependência a mais para instalar, e o código
deixa explícito exatamente o que uma chain faz por baixo (retriever →
prompt → llm → parser) — o que costuma ficar escondido atrás da
abstração de um framework. Trocar por `langchain.chains.RetrievalQA` é
uma troca de implementação, não de arquitetura: a assinatura
`RAGChain.run(question) -> RAGResult` continua a mesma.

**09. Fine-tuning (LoRA / QLoRA / PEFT)** (`src/finetune/lora_config.py`)
Este projeto usa RAG, não fine-tuning — decisão de design explicada no
próprio módulo: a base de conhecimento (políticas de suporte) muda com
frequência, e RAG permite atualizar a resposta do sistema editando um
arquivo `.md`, sem re-treinar nada. Fine-tuning faz mais sentido para
mudar o *estilo/comportamento* do modelo, não o conhecimento factual
dele. Ainda assim, o módulo contém a configuração real de LoRA
(`LoraConfig` da lib `peft`: rank, alpha, dropout, target_modules) que
seria usada caso o time decidisse migrar para uma abordagem híbrida
(RAG + modelo fine-tunado no tom de voz da empresa) — com um import
tardio da lib `peft`, para não exigir GPU/peft instalados só para rodar
o resto do projeto.

**10. Avaliação & Benchmarking** (`eval/evaluate.py`, `eval/eval_dataset.jsonl`)
Provavelmente o módulo mais importante do projeto do ponto de vista de
qualidade. O dataset de avaliação (`eval_dataset.jsonl`) tem 10
perguntas: 7 respondíveis (com uma keyword esperada na resposta, ex.:
"14 dias") e 3 propositalmente fora de domínio ("qual o preço de uma
pizza de calabresa?", "quem descobriu o Brasil?"). `evaluate()` roda
cada pergunta pelo pipeline RAG completo e calcula duas métricas:
- **Answer coverage**: das perguntas respondíveis, quantas o sistema
  respondeu corretamente (a keyword esperada apareceu na resposta)?
- **Hallucination rate**: das perguntas sem resposta na base, quantas
  o sistema respondeu como se soubesse, em vez de recusar?

Rodando neste repositório: **86% de coverage, 0% de hallucination**.
Essas métricas são o gate de qualidade usado no CI (etapa 14) — muito
mais relevante para um app de LLM do que "os testes unitários
passaram".

### Fase 03 — Engenharia & Deploy

**11. Docker** (`docker/Dockerfile`)
Build multi-stage, usuário não-root, `HEALTHCHECK` embutido — mesmo
padrão do projeto de MLOps. `ANTHROPIC_API_KEY` é passada como variável
de ambiente opcional; sem ela, o container roda com o provider mock.

**12. Model Serving (vLLM / TGI / Ollama)** (`docker/docker-compose.yml`,
`k8s/deployment.yaml`)
Como este projeto usa a API da Anthropic (ou o mock local), não há um
modelo local para servir com vLLM/TGI. O `docker-compose.yml` documenta,
comentado, como um serviço Ollama entraria no compose para servir um
modelo open-source localmente — e o `k8s/deployment.yaml` tem um
segundo `Deployment` (`llm-serving`) completo, com `nodeSelector` para
GPU e imagem `vllm/vllm-openai`, para o cenário em que o time decide
hospedar o próprio modelo em vez de usar uma API externa.

**13. Design de APIs (FastAPI)** (`api/main.py`)
Endpoint principal: `POST /chat`. O fluxo dentro do handler é: guardrails
de entrada (bloqueia com `400` se detectar prompt injection) → lookup no
cache (exato, depois semântico) → se não houver hit, roda a `RAGChain`
completa → guardrails de saída → grava no cache → retorna. A resposta
inclui `cache_hit` (`"exact"`, `"semantic"` ou `"miss"`), `sources`
(quais documentos foram usados) e `trace_id` — transparência total sobre
de onde veio cada resposta, essencial para debugar reclamações de "o
assistente respondeu errado".

**14. CI/CD para Apps de LLM** (`.github/workflows/ci-cd.yml`,
`tests/test_api.py`)
Três jobs: `test` (8 testes automatizados + lint), `eval_gate` (roda
`eval/evaluate.py` e falha o pipeline se `answer_coverage < 80%` ou
`hallucination_rate > 10%`), `build_and_push` (só roda se os dois
anteriores passarem). O `eval_gate` é o que diferencia CI/CD de LLM de
CI/CD tradicional: testes unitários garantem que o *código* não quebrou;
o eval gate garante que o *comportamento* do sistema (qualidade das
respostas) não regrediu.
Os 8 testes em `tests/test_api.py` cobrem: detecção de prompt injection,
perguntas normais não sendo bloqueadas, redação de PII, health check,
resposta correta a uma pergunta conhecida (contém "14 dias" e cita a
fonte certa), recusa correta de pergunta fora de domínio, bloqueio de
injection via API (`400`), e o ciclo completo de cache exato (primeira
chamada = miss, segunda chamada idêntica = exact hit) — todos passando.

**15. Guardrails & Segurança** (`guardrails/guardrails.py`)
Duas funções principais: `check_prompt_injection()` (regex contra
padrões conhecidos em PT/EN: "ignore as instruções anteriores",
"revele o system prompt", "modo desenvolvedor" etc. — bloqueia a
requisição) e `redact_pii()` (regex para CPF, e-mail e cartão de
crédito — redige automaticamente, sem bloquear, permitindo que a
pergunta continue sendo respondida com os dados sensíveis mascarados).
Testado com casos reais: uma tentativa de injection foi corretamente
bloqueada; uma pergunta com e-mail e CPF teve os dois campos redigidos
para `[E-MAIL REDIGIDO]` e `[CPF REDIGIDO]` automaticamente antes de
seguir para o resto do pipeline.

**16. Cache & Otimização de Custo** (`cache/semantic_cache.py`)
Duas camadas: cache **exato** (hash SHA-256 da pergunta normalizada —
pega perguntas idênticas, é essencialmente grátis em custo
computacional) e cache **semântico** (compara o embedding da nova
pergunta com os embeddings das perguntas já cacheadas via cosseno — pega
paráfrases). Cada entrada tem TTL (1 hora por padrão) e contador de
hits. Testado isoladamente: `"Quantos dias tenho para pedir reembolso?"`
seguido de `"Qual o prazo de dias para pedir reembolso integral?"`
resultou em `exact` na primeira chamada repetida e `semantic` na
paráfrase com overlap lexical suficiente — ver a seção de bug real
abaixo para uma limitação importante descoberta aqui.

**17. Observabilidade & Tracing** (`monitoring/tracing.py`)
A classe `Trace` gera um `trace_id` único por request e um `Span` por
etapa (`input_guardrails`, `cache_lookup`, `rag_chain`,
`output_guardrails`), cada um com duração em milissegundos e metadata.
`to_json()` serializa no formato que qualquer backend de observabilidade
real (LangSmith, Datadog, um dashboard Grafana alimentado por logs
estruturados) consumiria diretamente.

### Fase 04 — Escala & Governança

**18. Vector DBs em Escala** (`vectordb/README.md`)
Não é código executável — é um guia de decisão e migração, porque
"trocar o vector store" é uma decisão de infraestrutura que depende de
volume de dados e orçamento, não algo que faz sentido simular. Documenta
quando migrar (>100k chunks, múltiplos workers, filtros de metadata em
escala), compara Pinecone/Weaviate/Milvus/pgvector, e mostra o código de
adaptação exato (uma classe `PineconeVectorStore` com a mesma interface
`add`/`search` do `InMemoryVectorStore` atual) — a troca não exigiria
tocar em `rag_pipeline.py`.

**19. Cloud & Infra de GPU** (`infra/main.tf`, `k8s/deployment.yaml`)
Terraform com um node group de GPU dedicado (`g5.2xlarge`, 1x A10G) com
`taint`/`toleration` para garantir que só cargas de inferência de LLM
rodem ali — separado do node group da API (CPU-only), porque misturar
os dois desperdiça o recurso mais caro do cluster (a GPU) com tráfego
que não precisa dela.

**20. Governança, Compliance & System Design** (`governance/MODEL_CARD.md`,
`governance/SECURITY.md`)
O Model Card documenta a arquitetura (RAG, não fine-tuning, e por quê),
as métricas de avaliação com seus gates, e limitações conhecidas —
incluindo, de forma explícita, que o retriever TF-IDF é léxico, não
semântico. O `SECURITY.md` cobre riscos específicos de LLM: extração de
system prompt (tratado como superfície pública, nunca deve conter
segredos), PII em conversas, e **custo como vetor de ataque** — um
usuário malicioso pode gerar custo alto de API repetindo perguntas
caras, e o cache ajuda mas não substitui rate limiting por usuário/IP em
produção.

---

## Como rodar o projeto

```bash
make setup   # instala dependências (scikit-learn, fastapi, anthropic...)
make test    # roda os 8 testes automatizados
make eval    # avalia qualidade do RAG (coverage + hallucination rate)
make api     # sobe a API em http://localhost:8000
```

### Testar o chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quantos dias tenho para pedir reembolso?"}'
```

Resposta esperada: cita os 14 dias corridos, com `"sources":["politica_reembolso.md"]`
e `"cache_hit":"miss"`. Repita a mesma pergunta e observe `"cache_hit":"exact"`
na segunda resposta, com latência praticamente zero.

### Usar o provider real (Anthropic) em vez do mock

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make api
```

Nenhuma outra mudança de código é necessária — `src/llm_client.py`
detecta a chave automaticamente.

---

## Um bug real que apareceu enquanto construía isso

Vale documentar porque é exatamente o tipo de problema que "MLOps na
prática" é feito de resolver, não só de saber que existe em teoria.

Ao testar o pipeline com a pergunta fora de domínio "qual o preço de
uma pizza de calabresa?", o sistema **respondeu com um trecho da
política de reembolso** em vez de dizer que não sabia. A causa: o
retriever sempre devolve os `top_k` documentos mais similares, mesmo
quando nenhum deles é de fato relevante — sem um limiar de corte, "mais
similar entre os ruins" ainda parece uma resposta válida para o LLM.

A correção teve duas partes:
1. Adicionar um limiar mínimo de score (`min_score=0.1`) na `RAGChain` —
   documentos abaixo do limiar não entram no contexto.
2. Remover *stopwords* em português do TF-IDF. Sem isso, palavras de
   alta frequência (artigos, preposições) dominavam a similaridade de
   cosseno, e a pergunta da pizza tinha score artificialmente alto
   (~0.20) só por compartilhar palavras comuns como "de", "uma", "o" —
   quase no mesmo patamar de perguntas genuinamente relevantes. Depois
   da correção, a pergunta da pizza caiu para score 0.0 em todos os
   documentos, e o sistema passou a recusar corretamente.

Esse é o tipo de bug que só aparece testando com casos reais — por isso
o dataset de avaliação (`eval/eval_dataset.jsonl`) inclui de propósito
perguntas fora de domínio, não só perguntas que o sistema deveria
acertar.

---

## Decisões de design e trade-offs

- **TF-IDF em vez de embeddings densos reais**: permite que o projeto
  rode 100% offline, sem GPU, sem chamada de rede, sem custo. O
  trade-off, documentado explicitamente no Model Card e no
  `vectordb/README.md`, é que TF-IDF é *léxico* — só encontra
  paráfrases que compartilham palavras. Em produção, isso seria trocado
  por um embedder denso (`text-embedding-3-small`, Cohere, ou
  `sentence-transformers` local).
- **RAG em vez de fine-tuning**: justificado na etapa 09 acima — a base
  de conhecimento muda com frequência, e RAG permite atualização
  instantânea (editar um `.md`) sem re-treinar nada.
- **Provider mock determinístico, não um LLM pequeno local**: rodar um
  modelo pequeno local (ex.: via Ollama) ainda exigiria download de
  pesos (GBs) e mais tempo de inferência. O mock heurístico é
  suficiente para validar o *pipeline* (retrieval, guardrails, cache,
  tracing) — que é o que este projeto demonstra — sem confundir "o
  pipeline está certo" com "o modelo gera texto bonito".
- **Threshold do cache semântico calibrado para TF-IDF (0.35), não para
  embeddings densos (que ficariam em 0.85-0.92)**: documentado
  diretamente no código (`api/main.py`) para que a razão do número não
  fique perdida.

---

## Estratégia de testes e avaliação

Duas camadas complementares, cada uma pegando um tipo diferente de
regressão:
- **`tests/test_api.py`** (8 testes): testa *comportamento determinístico*
  — guardrails, contrato da API, cache. Roda em segundos, não depende de
  qualidade de geração de texto.
- **`eval/evaluate.py`**: testa *qualidade do sistema como um todo* —
  coverage e hallucination rate. É probabilístico por natureza (a
  resposta de um LLM real varia), por isso é medido como taxa sobre um
  dataset, não como assert binário por pergunta.

Ambos rodam via `make test` e `make eval`, e ambos são gates
independentes no CI (`.github/workflows/ci-cd.yml`).

---

## O que é real vs. o que é template

| Categoria | Status |
|---|---|
| Indexação, embeddings, busca vetorial, RAG chain | **Funcional** — testado, inclusive o bug de relevância documentado acima |
| Guardrails (injection, PII) | **Funcional** — testado com casos reais |
| Cache exato e semântico | **Funcional** — testado isoladamente e via API |
| API FastAPI + 8 testes | **Funcional** — testado com requests reais |
| Avaliação (coverage/hallucination) | **Funcional** — 86% coverage, 0% hallucination, reproduzível |
| Provider Anthropic real | **Código correto**, não executado neste ambiente (sem chave de API configurada aqui; segue o SDK oficial da Anthropic) |
| Docker, CI/CD (workflow YAML) | **Sintaticamente correto**, não executado (sem Docker/GitHub disponíveis neste ambiente) |
| Kubernetes, Terraform, Vector DB em escala, LoRA/PEFT | **Templates de produção** — configuração correta e comentada, dependem de GPU/cluster/conta cloud que não existem neste ambiente de geração |

---

## Troubleshooting

- **Servidor da API "morre" entre comandos de terminal**: se você rodar
  `uvicorn` em background num script/notebook, use `nohup` +
  detach completo do processo pai — processos em background presos ao
  shell que o iniciou são encerrados quando esse shell termina.
- **Cache semântico não pega uma paráfrase que "deveria" bater**:
  provavelmente a paráfrase não compartilha palavras suficientes com a
  pergunta original — é a limitação conhecida do TF-IDF (léxico, não
  semântico), documentada acima. Teste com um embedder denso real para
  comparar.
- **`UserWarning: Your stop_words may be inconsistent with your
  preprocessing`**: aviso do scikit-learn sobre normalização de acentos
  nas stopwords — inofensivo, mas resolvido usando a lista de stopwords
  já sem acentos em `src/embeddings.py`.

---

## Caminho para produção real

1. Trocar `src/embeddings.py` por um embedder denso real (API de
   embeddings ou `sentence-transformers` local) para retrieval
   semântico de verdade.
2. Migrar `InMemoryVectorStore` para Pinecone/Weaviate/Milvus quando a
   base de conhecimento crescer além do que cabe em memória — ver
   `vectordb/README.md`.
3. Configurar `ANTHROPIC_API_KEY` e comparar a qualidade das respostas
   do provider real vs. mock rodando `eval/evaluate.py` com os dois.
4. Adicionar rate limiting por usuário/IP na API antes de expor
   publicamente (mitigação de custo como vetor de ataque, ver
   `governance/SECURITY.md`).
5. Aplicar `k8s/deployment.yaml` num cluster real com node pool de GPU,
   se a decisão for hospedar o próprio modelo em vez de usar uma API
   externa.
