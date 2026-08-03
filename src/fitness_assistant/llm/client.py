"""Groq LLM client wrapper.

The client is instantiated once at module import time.  All call sites
use the ``llm()`` function rather than accessing the client directly.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> tuple[str, dict]:
    """Send a single-turn prompt to Groq and return the response.

    Args:
        prompt: The complete prompt string to send.
        model: Groq model identifier (default: llama-3.3-70b-versatile).

    Returns:
        A tuple of ``(response_text, token_stats)`` where ``token_stats`` is a
        dict with keys ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
    """
    chat_completion = _client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    token_stats = {
        "prompt_tokens": chat_completion.usage.prompt_tokens,
        "completion_tokens": chat_completion.usage.completion_tokens,
        "total_tokens": chat_completion.usage.total_tokens,
    }
    return chat_completion.choices[0].message.content, token_stats