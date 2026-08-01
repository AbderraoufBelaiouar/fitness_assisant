import json
from functools import lru_cache
from pathlib import Path

from minsearch import Index

TEXT_FIELDS = [
    "name",
    "category",
    "body_part",
    "equipment",
    "muscle_group",
    "target",
    "secondary_muscles",
    "instructions",
    "instruction_steps",
]

KEYWORD_FIELDS = ["id", "media_id"]

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "exercises_prepared.json"
)


def build_index(
    documents: list[dict],
    text_fields: list[str] | None = None,
    keyword_fields: list[str] | None = None,
) -> Index:
    index = Index(
        text_fields=text_fields or TEXT_FIELDS,
        keyword_fields=keyword_fields or KEYWORD_FIELDS,
    )
    index.fit(documents)
    return index


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    with open(PROCESSED_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_index() -> Index:
    return build_index(load_documents())
