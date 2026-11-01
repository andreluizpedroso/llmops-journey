# LLMOps Journey — RAG Support Assistant

[![CI/CD](https://github.com/andreluizpedroso/llmops-journey/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/andreluizpedroso/llmops-journey/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Anthropic Claude](https://img.shields.io/badge/LLM-Anthropic%20Claude-D97757)

🇧🇷 [Leia em português](README.pt-BR.md)

An end-to-end project built by following the **20 steps** of the
*"The LLMOps Journey"* roadmap. **It runs 100% offline, at no cost and
with no API key** by default, using a deterministic local mock LLM
provider. It switches to the real Anthropic provider with a single
environment variable, without changing a line of code.

**Chosen use case:** a support assistant that answers questions about the
policies of a fictional company (Aurora Cloud: refunds, usage limits,
security, SLA) using **RAG** (retrieval-augmented generation). This is the
most common LLM app pattern in production today, and it was chosen on
purpose because it exposes the real LLMOps problems (hallucination, cost,
latency, security), not just "calling an LLM API".

---

## Table of Contents

1. [How the Pieces Fit Together](#how-the-pieces-fit-together)
2. [Step by Step: What Was Built and Why](#step-by-step-what-was-built-and-why)
3. [How to Run the Project](#how-to-run-the-project)
4. [A Real Bug That Showed Up While Building This](#a-real-bug-that-showed-up-while-building-this)
5. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
6. [Testing and Evaluation Strategy](#testing-and-evaluation-strategy)
7. [What Is Real vs. What Is a Template](#what-is-real-vs-what-is-a-template)
8. [Troubleshooting](#troubleshooting)
9. [Path to Real Production](#path-to-real-production)

---

## How the Pieces Fit Together

```
data/knowledge_base/*.md          src/prompts.py (v1/v2)
        │  source documents              │  versioned templates
        ▼                                │
src/indexer.py  ── chunking + indexing   │
        │                                │
        ▼                                │
src/embeddings.py (TF-IDF)               │
        │                                │
        ▼                                │
src/vector_store.py (cosine search)      │
        │                                │
        └──────────► src/rag_pipeline.py ◄┘
                            │
                            ▼
                    src/llm_client.py
                    (real Anthropic OR local mock)
                            │
                            ▼
                      api/main.py (FastAPI)
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
guardrails/            cache/                monitoring/
guardrails.py          semantic_cache.py     tracing.py
(injection + PII)      (exact + semantic)    (spans per request)
        │
        ▼
tests/test_api.py  ──  eval/evaluate.py  ──  .github/workflows/ci-cd.yml
(8 tests)               (coverage/hallucination)  (quality gates)
```

A user request passes, in order, through: **input guardrails** (block
prompt injection, redact PII) → **cache** (exact, then semantic) → on a
miss, the **full RAG chain** (retrieval → prompt → generation) →
**output guardrails** → response. Each stage on this path becomes a
tracing *span*, visible through the `trace_id` returned in the response.

---

## Step by Step: What Was Built and Why

### Phase 01 — Fundamentals

**01. Python** — all of the project's logic.

**02. Prompt Engineering** (`src/prompts.py`)
Prompts are treated as code: versioned (`v1`, `v2`) in a
`PROMPT_VERSIONS` dictionary, not as loose strings scattered across the
project. `v1` is the basic prompt ("answer only from the context; say you
don't know if the information is missing"). `v2` adds an extra
instruction asking for concise one-sentence answers that cite the exact
number/deadline. In a real project, that iteration would come from
observing evasive answers in `eval/evaluate.py`. `build_prompt()` is the
function any part of the system calls to assemble the final prompt;
switching versions in production (for A/B comparison) means passing a
parameter, not editing strings in production.

**03. NLP Fundamentals** (`src/embeddings.py`)
It has a `tokenize_preview()` function solely to illustrate the concept
of tokenization (a real LLM uses BPE/SentencePiece, not whitespace
splitting). The part that actually does the work is the `Embedder`, based
on scikit-learn's TF-IDF. This was a deliberate decision to run offline
with no GPU and no network calls (see the sections below on why this is
lexical, not semantic).

**04. Git & Version Control** (`.gitignore`)
Ignores secrets (`ANTHROPIC_API_KEY`) and cache/test artifacts.

**05. LLM APIs** (`src/llm_client.py`)
This is one of the project's core modules. It defines an abstract
`LLMClient` interface with a single method `generate(system, user,
max_tokens) -> LLMResponse`, and two implementations:
- `AnthropicClient`: calls the real Anthropic API (`claude-sonnet-4-6`),
  used when `ANTHROPIC_API_KEY` is set in the environment.
- `MockLLMClient`: a local, deterministic, free provider. Instead of
  generating free text, it extracts from the `<context>` the sentence(s)
  with the greatest word overlap with the `<question>`. This is a simple
  heuristic, but enough to test the **entire** pipeline (retrieval →
  prompt → generation → guardrails → cache → tracing) at no cost and
  without depending on the network.

The `get_llm_client()` function is a factory: it picks the provider
automatically by checking whether the environment variable exists. It is
the only decision point in the whole project. Everything else (RAG
chain, API) neither knows nor needs to know which provider is active.

### Phase 02 — Building with LLMs

**06. Embeddings & Vector Search** (`src/embeddings.py`, `src/vector_store.py`)
`Embedder.fit()` trains a `TfidfVectorizer` (unigrams + bigrams, up to
4096 features) over the knowledge base documents.
`InMemoryVectorStore` keeps the vectors in a numpy matrix and performs
vectorized cosine-similarity search (`search(query_vector, k)`).
The interface (`add`, `search`) is designed to be exactly the one a real
vector DB (Pinecone, Weaviate) would implement. See `vectordb/README.md`
for the migration guide.

**07. RAG** (`src/indexer.py`, `src/rag_pipeline.py`)
`indexer.py` reads the knowledge base `.md` files, chunks them by
character count with overlap (400 characters, 80 of overlap, which avoids
cutting an important sentence right at the boundary between two chunks),
and indexes them in the vector store. `rag_pipeline.py` implements the
`RAGChain`: it receives a question, retrieves the `top_k=3` most similar
chunks, **filters by a minimum relevance threshold** (`min_score=0.1`),
builds the prompt with the filtered context, and calls the LLM. The
relevance filter is what stops the system from trying to answer
out-of-domain questions. See the real-bug section below: this was not a
theoretical decision, it was a problem that surfaced and was fixed while
building this project.

**08. Frameworks (LangChain / LlamaIndex)** (`src/rag_pipeline.py`)
`RAGChain` implements by hand the "chain" pattern that frameworks such as
LangChain (`RetrievalQA`) or LlamaIndex (`QueryEngine`) provide out of the
box. Not using a heavyweight framework here was deliberate: the project
has one fewer dependency to install, and the code makes explicit exactly
what a chain does under the hood (retriever → prompt → llm → parser),
which is usually hidden behind a framework's abstraction. Swapping in
`langchain.chains.RetrievalQA` is a change of implementation, not of
architecture: the `RAGChain.run(question) -> RAGResult` signature stays
the same.

**09. Fine-tuning (LoRA / QLoRA / PEFT)** (`src/finetune/lora_config.py`)
This project uses RAG, not fine-tuning. The design decision is explained
in the module itself: the knowledge base (support policies) changes
frequently, and RAG lets you update the system's answers by editing a
`.md` file, with no retraining. Fine-tuning makes more sense for changing
the model's *style/behavior*, not its factual knowledge. Even so, the
module contains the real LoRA configuration (`LoraConfig` from the `peft`
library: rank, alpha, dropout, target_modules) that would be used if the
team decided to move to a hybrid approach (RAG + a model fine-tuned on the
company's tone of voice). It uses a late import of `peft`, so that no GPU
or `peft` installation is required just to run the rest of the project.

**10. Evaluation & Benchmarking** (`eval/evaluate.py`, `eval/eval_dataset.jsonl`)
Probably the most important module in the project from a quality
standpoint. The evaluation dataset (`eval_dataset.jsonl`) has 10
questions: 7 answerable (each with an expected keyword in the answer,
e.g. "14 dias") and 3 deliberately out of domain ("qual o preço de uma
pizza de calabresa?" / "how much is a pepperoni pizza?", "quem descobriu
o Brasil?" / "who discovered Brazil?"). `evaluate()` runs each question
through the full RAG pipeline and computes two metrics:
- **Answer coverage**: of the answerable questions, how many did the
  system answer correctly (the expected keyword appeared in the answer)?
- **Hallucination rate**: of the questions with no answer in the
  knowledge base, how many did the system answer as if it knew, instead
  of refusing?

Running on this repository: **86% coverage, 0% hallucination**.
These metrics are the quality gate used in CI (step 14), far more
relevant for an LLM app than "the unit tests passed".

### Phase 03 — Engineering & Deployment

**11. Docker** (`docker/Dockerfile`)
Multi-stage build, non-root user, built-in `HEALTHCHECK`, the same
pattern as my MLOps project. `ANTHROPIC_API_KEY` is passed as an optional
environment variable; without it, the container runs with the mock
provider.

**12. Model Serving (vLLM / TGI / Ollama)** (`docker/docker-compose.yml`,
`k8s/deployment.yaml`)
Since this project uses the Anthropic API (or the local mock), there is
no local model to serve with vLLM/TGI. `docker-compose.yml` documents, in
comments, how an Ollama service would be added to the compose file to
serve an open-source model locally. `k8s/deployment.yaml` has a second,
complete `Deployment` (`llm-serving`), with a GPU `nodeSelector` and the
`vllm/vllm-openai` image, for the scenario where the team decides to host
its own model instead of using an external API.

**13. API Design (FastAPI)** (`api/main.py`)
Main endpoint: `POST /chat`. The flow inside the handler is: input
guardrails (blocks with `400` if prompt injection is detected) → cache
lookup (exact, then semantic) → on a miss, run the full `RAGChain` →
output guardrails → write to cache → return. The response includes
`cache_hit` (`"exact"`, `"semantic"` or `"miss"`), `sources` (which
documents were used) and `trace_id`, giving full transparency into where
each answer came from, which is essential for debugging "the assistant
answered wrong" complaints.

**14. CI/CD for LLM Apps** (`.github/workflows/ci-cd.yml`,
`tests/test_api.py`)
Three jobs: `test` (8 automated tests + lint), `eval_gate` (runs
`eval/evaluate.py` and fails the pipeline if `answer_coverage < 80%` or
`hallucination_rate > 10%`), and `build_and_push` (runs only if the
previous two pass). The `eval_gate` is what sets LLM CI/CD apart from
traditional CI/CD: unit tests ensure the *code* didn't break; the eval
gate ensures the system's *behavior* (answer quality) didn't regress.
The 8 tests in `tests/test_api.py` cover: prompt injection detection,
normal questions not being blocked, PII redaction, health check, a
correct answer to a known question (contains "14 dias" and cites the
right source), correct refusal of an out-of-domain question, injection
blocking via the API (`400`), and the full exact-cache cycle (first call
= miss, second identical call = exact hit). All pass.

**15. Guardrails & Security** (`guardrails/guardrails.py`)
Two main functions: `check_prompt_injection()` (regex against known
patterns in PT/EN: "ignore as instruções anteriores" (ignore previous
instructions), "revele o system prompt" (reveal the system prompt),
"modo desenvolvedor" (developer mode), etc.; it blocks the request) and
`redact_pii()` (regex for CPF, e-mail and credit card; it redacts
automatically without blocking, so the question can still be answered
with the sensitive data masked). Tested with real cases: an injection
attempt was correctly blocked; a question containing an e-mail and a CPF
had both fields automatically redacted to `[E-MAIL REDIGIDO]` and
`[CPF REDIGIDO]` before moving on to the rest of the pipeline.

**16. Cache & Cost Optimization** (`cache/semantic_cache.py`)
Two layers: an **exact** cache (SHA-256 hash of the normalized question;
catches identical questions and is essentially free computationally) and
a **semantic** cache (compares the new question's embedding with the
embeddings of already-cached questions via cosine similarity; catches
paraphrases). Each entry has a TTL (1 hour by default) and a hit counter.
Tested in isolation: `"Quantos dias tenho para pedir reembolso?"`
followed by `"Qual o prazo de dias para pedir reembolso integral?"`
resulted in `exact` on the first repeated call and `semantic` on the
paraphrase with enough lexical overlap. See the real-bug section below
for an important limitation discovered here.

**17. Observability & Tracing** (`monitoring/tracing.py`)
The `Trace` class generates a unique `trace_id` per request and a `Span`
per stage (`input_guardrails`, `cache_lookup`, `rag_chain`,
`output_guardrails`), each with a duration in milliseconds and metadata.
`to_json()` serializes into the format that any real observability
backend (LangSmith, Datadog, a Grafana dashboard fed by structured logs)
would consume directly.

### Phase 04 — Scale & Governance

**18. Vector DBs at Scale** (`vectordb/README.md`)
This is not executable code. It is a decision and migration guide,
because "swapping the vector store" is an infrastructure decision that
depends on data volume and budget, not something that makes sense to
simulate. It documents when to migrate (>100k chunks, multiple workers,
metadata filters at scale), compares Pinecone/Weaviate/Milvus/pgvector,
and shows the exact adaptation code (a `PineconeVectorStore` class with
the same `add`/`search` interface as the current `InMemoryVectorStore`).
The swap would not require touching `rag_pipeline.py`.

**19. Cloud & GPU Infrastructure** (`infra/main.tf`, `k8s/deployment.yaml`)
Terraform with a dedicated GPU node group (`g5.2xlarge`, 1x A10G) with a
`taint`/`toleration` to ensure that only LLM inference workloads run
there. It is kept separate from the API node group (CPU-only), because
mixing the two wastes the cluster's most expensive resource (the GPU) on
traffic that doesn't need it.

**20. Governance, Compliance & System Design** (`governance/MODEL_CARD.md`,
`governance/SECURITY.md`)
The Model Card documents the architecture (RAG, not fine-tuning, and
why), the evaluation metrics with their gates, and known limitations,
including, explicitly, that the TF-IDF retriever is lexical, not
semantic. `SECURITY.md` covers LLM-specific risks: system prompt
extraction (treated as a public surface that must never contain secrets),
PII in conversations, and **cost as an attack vector**: a malicious user
can generate high API costs by repeating expensive questions, and the
cache helps but does not replace per-user/IP rate limiting in production.

---

## How to Run the Project

```bash
make setup   # installs dependencies (scikit-learn, fastapi, anthropic...)
make test    # runs the 8 automated tests
make eval    # evaluates RAG quality (coverage + hallucination rate)
make api     # starts the API at http://localhost:8000
```

### Try the chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quantos dias tenho para pedir reembolso?"}'
```

Expected response: cites the 14 calendar days, with `"sources":["politica_reembolso.md"]`
and `"cache_hit":"miss"`. Repeat the same question and observe `"cache_hit":"exact"`
in the second response, with practically zero latency.

### Use the real provider (Anthropic) instead of the mock

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make api
```

No other code change is needed; `src/llm_client.py` detects the key
automatically.

---

## A Real Bug That Showed Up While Building This

Worth documenting, because it is exactly the kind of problem that
"MLOps in practice" is made of solving, not just knowing it exists in
theory.

When testing the pipeline with the out-of-domain question "qual o preço
de uma pizza de calabresa?" ("how much is a pepperoni pizza?"), the
system **answered with an excerpt from the refund policy** instead of
saying it didn't know. The cause: the retriever always returns the
`top_k` most similar documents, even when none of them is actually
relevant. Without a cutoff threshold, "most similar among the bad ones"
still looks like a valid answer to the LLM.

The fix had two parts:
1. Add a minimum score threshold (`min_score=0.1`) to `RAGChain`;
   documents below the threshold do not enter the context.
2. Remove Portuguese *stopwords* from the TF-IDF. Without this,
   high-frequency words (articles, prepositions) dominated the cosine
   similarity, and the pizza question had an artificially high score
   (~0.20) just from sharing common words such as "de", "uma", "o",
   nearly on par with genuinely relevant questions. After the fix, the
   pizza question dropped to a score of 0.0 across all documents, and the
   system started refusing correctly.

This is the kind of bug that only shows up when testing with real cases,
which is why the evaluation dataset (`eval/eval_dataset.jsonl`)
deliberately includes out-of-domain questions, not just questions the
system should get right.

---

## Design Decisions and Trade-offs

- **TF-IDF instead of real dense embeddings**: lets the project run 100%
  offline, with no GPU, no network calls, and no cost. The trade-off,
  explicitly documented in the Model Card and in `vectordb/README.md`, is
  that TF-IDF is *lexical*: it only finds paraphrases that share words.
  In production, this would be replaced by a dense embedder
  (`text-embedding-3-small`, Cohere, or a local `sentence-transformers`).
- **RAG instead of fine-tuning**: justified in step 09 above. The
  knowledge base changes frequently, and RAG allows instant updates
  (editing a `.md`) without retraining.
- **Deterministic mock provider, not a small local LLM**: running a small
  local model (e.g. via Ollama) would still require downloading weights
  (GBs) and more inference time. The heuristic mock is enough to validate
  the *pipeline* (retrieval, guardrails, cache, tracing), which is what
  this project demonstrates, without conflating "the pipeline is correct"
  with "the model generates nice text".
- **Semantic cache threshold calibrated for TF-IDF (0.35), not for dense
  embeddings (which would sit at 0.85-0.92)**: documented directly in the
  code (`api/main.py`) so the reason for the number doesn't get lost.

---

## Testing and Evaluation Strategy

Two complementary layers, each catching a different kind of regression:
- **`tests/test_api.py`** (8 tests): tests *deterministic behavior*
  (guardrails, API contract, cache). Runs in seconds and does not depend
  on text generation quality.
- **`eval/evaluate.py`**: tests the *quality of the system as a whole*
  (coverage and hallucination rate). It is probabilistic by nature (a real
  LLM's answers vary), so it is measured as a rate over a dataset, not as
  a binary assert per question.

Both run via `make test` and `make eval`, and both are independent gates
in CI (`.github/workflows/ci-cd.yml`).

---

## What Is Real vs. What Is a Template

| Category | Status |
|---|---|
| Indexing, embeddings, vector search, RAG chain | **Functional**: tested, including the relevance bug documented above |
| Guardrails (injection, PII) | **Functional**: tested with real cases |
| Exact and semantic cache | **Functional**: tested in isolation and via the API |
| FastAPI API + 8 tests | **Functional**: tested with real requests |
| Evaluation (coverage/hallucination) | **Functional**: 86% coverage, 0% hallucination, reproducible |
| Real Anthropic provider | **Correct code**, not executed in this environment (no API key configured here; follows the official Anthropic SDK) |
| Docker, CI/CD (workflow YAML) | **Syntactically correct**, not executed (no Docker/GitHub available in this environment) |
| Kubernetes, Terraform, Vector DB at scale, LoRA/PEFT | **Production templates**: correct, commented configuration that depends on a GPU/cluster/cloud account that do not exist in this generation environment |

---

## Troubleshooting

- **API server "dies" between terminal commands**: if you run `uvicorn`
  in the background from a script/notebook, use `nohup` plus full
  detachment from the parent process. Background processes tied to the
  shell that started them are terminated when that shell exits.
- **Semantic cache doesn't catch a paraphrase that "should" match**: the
  paraphrase most likely doesn't share enough words with the original
  question. This is the known limitation of TF-IDF (lexical, not
  semantic), documented above. Test with a real dense embedder to
  compare.
- **`UserWarning: Your stop_words may be inconsistent with your
  preprocessing`**: a scikit-learn warning about accent normalization in
  the stopwords. It is harmless, and was resolved by using the
  already-unaccented stopword list in `src/embeddings.py`.

---

## Path to Real Production

1. Replace `src/embeddings.py` with a real dense embedder (embeddings API
   or a local `sentence-transformers`) for truly semantic retrieval.
2. Migrate `InMemoryVectorStore` to Pinecone/Weaviate/Milvus when the
   knowledge base grows beyond what fits in memory. See
   `vectordb/README.md`.
3. Set `ANTHROPIC_API_KEY` and compare answer quality of the real provider
   vs. the mock by running `eval/evaluate.py` with both.
4. Add per-user/IP rate limiting to the API before exposing it publicly
   (mitigating cost as an attack vector, see `governance/SECURITY.md`).
5. Apply `k8s/deployment.yaml` to a real cluster with a GPU node pool, if
   the decision is to host the model yourself instead of using an
   external API.
