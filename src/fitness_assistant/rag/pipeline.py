"""RAG pipeline: retrieve → prompt → generate → evaluate.

The pipeline runs the full cycle in one call and returns a single,
self-contained result dict that is ready to be persisted to the database.
"""

from time import time

from eval.llm_judge import evaluate_relevance
from llm.client import llm
from llm.prompts import build_prompt
from search.search import search


def rag(query: str, index, model: str = "llama-3.3-70b-versatile") -> dict:
    """Run a full RAG cycle (retrieve → generate → self-evaluate).

    Args:
        query: The user's natural-language question.
        index: A fitted minsearch Index.
        model: Groq model identifier to use for generation and evaluation.

    Returns:
        A dict containing:
            answer, model_used, response_time,
            relevance, relevance_explanation,
            prompt_tokens, completion_tokens, total_tokens,
            eval_prompt_tokens, eval_completion_tokens, eval_total_tokens,
            groq_cost (always 0.0 — Groq's free tier has no per-token charge).
    """
    t0 = time()
    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)
    response_time = time() - t0

    evaluation, eval_token_stats = evaluate_relevance(query, answer, model=model)

    return {
        "answer": answer,
        "model_used": model,
        "response_time": round(response_time, 3),
        "relevance": evaluation.get("Relevance", "UNKNOWN"),
        "relevance_explanation": evaluation.get("Explanation", ""),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": eval_token_stats["prompt_tokens"],
        "eval_completion_tokens": eval_token_stats["completion_tokens"],
        "eval_total_tokens": eval_token_stats["total_tokens"],
        # Groq is currently free; field kept for schema compatibility.
        "groq_cost": 0.0,
    }