"""LLM-as-judge utilities for relevance evaluation.

These helpers are shared by the RAG evaluation pipeline.  The module has
no side-effects at import time — all state lives inside functions.
"""

import json
import re
import time

from llm.client import llm

_EVAL_MODEL = "llama-3.3-70b-versatile"
_MAX_RETRIES = 3
_VALID_RELEVANCE = {"NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"}

_EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as 'NON_RELEVANT', 'PARTLY_RELEVANT', or 'RELEVANT'.

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def _extract_evaluation(response: str) -> dict:
    """Parse a JSON evaluation from an LLM response string."""
    text = response.strip()
    # Strip optional markdown code fences.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: replace single-quoted keys/values with double quotes.
        text = re.sub(r"(?<=\W)'|'(?=\W)", '"', text)
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM evaluation response is not a JSON object")
    relevance = data.get("Relevance")
    if relevance not in _VALID_RELEVANCE:
        raise ValueError(f"Invalid relevance value from LLM judge: {relevance!r}")
    return data


def evaluate_relevance(
    question: str,
    answer: str,
    model: str = _EVAL_MODEL,
    max_retries: int = _MAX_RETRIES,
) -> tuple[dict, dict]:
    """Ask an LLM judge to rate the relevance of *answer* to *question*.

    Returns:
        A tuple of (evaluation_dict, token_stats_dict).
        evaluation_dict has keys ``Relevance`` and ``Explanation``.
        On repeated failures, returns ``{"Relevance": "UNKNOWN", ...}`` with
        zero token counts rather than raising, so callers can continue a batch.
    """
    prompt = _EVALUATION_PROMPT_TEMPLATE.format(question=question, answer=answer)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response_text, token_stats = llm(prompt, model=model)
            evaluation = _extract_evaluation(response_text)
            return evaluation, token_stats
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 ** attempt)  # exponential back-off: 1 s, 2 s, 4 s

    # Graceful degradation: log the failure but do not crash the caller.
    return (
        {"Relevance": "UNKNOWN", "Explanation": f"Failed to parse: {last_error}"},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )