"""Dataset loading and preparation for the fitness assistant.

Running this module directly (re-)generates the prepared document cache:

    uv run python src/fitness_assistant/ingestion/load.py
"""

from pathlib import Path

import pandas as pd

from ingestion.documents import prepare_documents

# ---------------------------------------------------------------------------
# Canonical paths — resolved relative to this file so the module works
# regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "exercises.json"
PREPARED_DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "exercises_prepared.json"


def load_exercises(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PREPARED_DATA_PATH,
) -> list[dict]:
    """Load, prepare, and cache the exercise dataset.

    Args:
        raw_path: Location of the source ``exercises.json`` file.
        output_path: Where to write the prepared (cached) dataset.

    Returns:
        A list of prepared exercise dicts.
    """
    raw_documents: list[dict] = pd.read_json(raw_path).to_dict(orient="records")
    docs = prepare_documents(raw_documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(docs).to_json(output_path, orient="records", indent=4)
    return docs
