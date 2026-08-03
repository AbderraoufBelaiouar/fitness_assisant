"""RAG pipeline: retrieve → prompt → generate."""

from time import time

from llm.client import llm
from llm.prompts import build_prompt
from search.search import search


def rag(query: str, index, model: str = "llama-3.3-70b-versatile") -> dict:
    """Run a full RAG cycle and return a structured result dict.

    Args:
        query: The user's natural-language question.
        index: A fitted minsearch Index.
        model: Groq model identifier to use for generation.

    Returns:
        A dict containing the answer, timing, token usage, and the model used.
    """
    t0 = time()
    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)
    response_time = time() - t0

    return {
        "answer": answer,
        "model_used": model,
        "response_time": round(response_time, 3),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
    }