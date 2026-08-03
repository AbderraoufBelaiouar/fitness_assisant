"""Fitness-assistant Flask API with conversation logging and feedback.

Endpoints:
    GET  /health      – liveness probe
    POST /question    – answer a fitness question via RAG
    POST /feedback    – record thumbs-up (+1) or thumbs-down (-1)

Environment:
    See db.py for database connection variables.
    GROQ_API_KEY must be set for LLM calls.
"""

import uuid

from flask import Flask, jsonify, request

import db
from rag.pipeline import rag
from search.index import get_index

app = Flask(__name__)

# Build and cache the keyword-search index once at startup.
_index = get_index()


@app.get("/health")
def health() -> tuple:
    """Liveness probe — returns 200 when the service is ready."""
    return jsonify({"status": "ok"}), 200


@app.post("/question")
def handle_question() -> tuple:
    """Answer a fitness question and log the full conversation.

    Request body (JSON):
        {"question": "<user question>"}

    Response body (JSON):
        {
          "conversation_id": "<uuid>",
          "question": "<echo of input>",
          "answer": "<LLM answer>",
          "model_used": "<model name>",
          "response_time": <float seconds>,
          "relevance": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT",
          "prompt_tokens": <int>,
          "completion_tokens": <int>,
          "total_tokens": <int>
        }
    """
    data = request.get_json(silent=True) or {}
    question: str = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Field 'question' is required and must not be blank."}), 400

    conversation_id = str(uuid.uuid4())
    answer_data = rag(question, _index)

    db.save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer_data=answer_data,
    )

    return (
        jsonify(
            {
                "conversation_id": conversation_id,
                "question": question,
                "answer": answer_data["answer"],
                "model_used": answer_data["model_used"],
                "response_time": answer_data["response_time"],
                "relevance": answer_data["relevance"],
                "prompt_tokens": answer_data["prompt_tokens"],
                "completion_tokens": answer_data["completion_tokens"],
                "total_tokens": answer_data["total_tokens"],
            }
        ),
        200,
    )


@app.post("/feedback")
def handle_feedback() -> tuple:
    """Record user feedback for a completed conversation.

    Request body (JSON):
        {
          "conversation_id": "<uuid returned by /question>",
          "feedback": 1   -- thumbs up
          OR
          "feedback": -1  -- thumbs down
        }
    """
    data = request.get_json(silent=True) or {}
    conversation_id: str = (data.get("conversation_id") or "").strip()
    feedback = data.get("feedback")

    if not conversation_id:
        return jsonify({"error": "Field 'conversation_id' is required."}), 400
    if feedback not in (1, -1):
        return jsonify({"error": "Field 'feedback' must be 1 (positive) or -1 (negative)."}), 400

    db.save_feedback(conversation_id=conversation_id, feedback=feedback)
    return jsonify({"message": f"Feedback received: {feedback}"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
