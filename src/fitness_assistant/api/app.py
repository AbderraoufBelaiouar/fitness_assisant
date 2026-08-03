"""Fitness-assistant Flask API.

Start the development server:
    uv run python -m fitness_assistant.api.app

Or via the package entrypoint (if configured in pyproject.toml):
    uv run fitness-api
"""

import uuid

from flask import Flask, jsonify, request

from rag.pipeline import rag
from search.index import get_index

app = Flask(__name__)

# Build and cache the search index once at startup.
_index = get_index()


@app.get("/health")
def health() -> tuple:
    """Liveness probe — returns 200 when the service is ready."""
    return jsonify({"status": "ok"}), 200


@app.post("/question")
def handle_question() -> tuple:
    """Answer a fitness question using the RAG pipeline.

    Request body (JSON):
        {"question": "<user question>"}

    Response body (JSON):
        {
          "conversation_id": "<uuid>",
          "question": "<echo of input>",
          "answer": "<LLM answer string>",
          "model_used": "<model name>",
          "response_time": <seconds (float)>,
          "prompt_tokens": <int>,
          "completion_tokens": <int>,
          "total_tokens": <int>
        }
    """
    data = request.get_json(silent=True) or {}
    question: str = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Field 'question' is required and must not be blank."}), 400

    rag_result = rag(question, _index)

    return (
        jsonify(
            {
                "conversation_id": str(uuid.uuid4()),
                "question": question,
                **rag_result,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
