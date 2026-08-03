"""End-to-end RAG answer-relevance evaluation.

Usage:
    uv run python src/fitness_assistant/eval/evaluate_rag.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the src/ layout importable when running this script directly.
# ---------------------------------------------------------------------------
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import pandas as pd
from tqdm.auto import tqdm

from eval.llm_judge import evaluate_relevance
from load_ground_truth import load_ground_truth
from rag.pipeline import rag
from search.index import get_index

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "rag-evaluations.csv"

MODEL = "llama-3.3-70b-versatile"


def evaluate_record(record: dict, index) -> dict:
    """Run RAG on one ground-truth record and judge the answer.

    Returns a flat dict suitable for a DataFrame row.
    """
    question = record["question"]
    rag_result = rag(question, index, model=MODEL)
    answer = rag_result["answer"]

    evaluation, eval_token_stats = evaluate_relevance(question, answer, model=MODEL)

    return {
        "id": record["id"],
        "question": question,
        "answer": answer,
        "relevance": evaluation.get("Relevance", "UNKNOWN"),
        "explanation": evaluation.get("Explanation", ""),
        "response_time": rag_result["response_time"],
        "prompt_tokens": rag_result["prompt_tokens"],
        "completion_tokens": rag_result["completion_tokens"],
        "total_tokens": rag_result["total_tokens"],
        "eval_prompt_tokens": eval_token_stats["prompt_tokens"],
        "eval_completion_tokens": eval_token_stats["completion_tokens"],
        "eval_total_tokens": eval_token_stats["total_tokens"],
    }


def main() -> None:
    sample = load_ground_truth()
    index = get_index()

    rows: list[dict] = []
    for record in tqdm(sample, desc="Evaluating RAG"):
        try:
            rows.append(evaluate_record(record, index))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping record {record.get('id')!r}: {exc}", file=sys.stderr)

    df = pd.DataFrame(rows)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"\nWrote {len(df)} evaluation rows to {DEFAULT_OUTPUT}")
    print(df["relevance"].value_counts(normalize=True).to_string())


if __name__ == "__main__":
    main()
