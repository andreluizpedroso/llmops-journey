"""
Etapa 10 — Avaliação & Benchmarking
----------------------------------------
Roda o dataset de avaliação contra o pipeline RAG e mede:
  - Answer coverage: perguntas respondíveis que de fato receberam
    a informação certa (keyword esperada aparece na resposta).
  - Hallucination rate: perguntas SEM resposta na base que o sistema
    respondeu como se tivesse (deveria ter recusado).
  - Latência média.

Isso serve tanto como regressão de qualidade no CI (etapa 14) quanto
para comparar versões de prompt (v1 vs v2 — ver src/prompts.py).
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from indexer import build_index  # noqa: E402
from rag_pipeline import RAGChain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "eval_dataset.jsonl"

REFUSAL_MARKERS = ["não encontrei", "nao encontrei", "abrir um ticket"]


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate(prompt_version: str = "v2") -> dict:
    store, embedder = build_index()
    chain = RAGChain(store, embedder, prompt_version=prompt_version)
    dataset = load_dataset()

    answerable = [ex for ex in dataset if ex["expects_answer"]]
    unanswerable = [ex for ex in dataset if not ex["expects_answer"]]

    correct_answers = 0
    latencies = []
    details = []

    for ex in answerable:
        result = chain.run(ex["question"])
        latencies.append(result.llm_response.latency_ms)
        hit = any(kw.lower() in result.answer.lower() for kw in ex["expected_keywords"])
        correct_answers += int(hit)
        details.append({"question": ex["question"], "hit": hit, "answer": result.answer[:80]})

    hallucinations = 0
    for ex in unanswerable:
        result = chain.run(ex["question"])
        latencies.append(result.llm_response.latency_ms)
        refused = is_refusal(result.answer)
        hallucinations += int(not refused)
        details.append(
            {"question": ex["question"], "correctly_refused": refused, "answer": result.answer[:80]}
        )

    return {
        "prompt_version": prompt_version,
        "answer_coverage": round(correct_answers / len(answerable), 3) if answerable else None,
        "hallucination_rate": round(hallucinations / len(unanswerable), 3) if unanswerable else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "details": details,
    }


if __name__ == "__main__":
    results = evaluate(prompt_version="v2")
    print(f"Prompt version:     {results['prompt_version']}")
    print(f"Answer coverage:    {results['answer_coverage']:.0%}  ({results['n_answerable']} perguntas respondíveis)")
    print(f"Hallucination rate: {results['hallucination_rate']:.0%}  ({results['n_unanswerable']} perguntas sem resposta na base)")
    print(f"Latência média:     {results['avg_latency_ms']} ms")
    print()
    for d in results["details"]:
        print(d)
