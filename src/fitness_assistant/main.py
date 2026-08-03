"""Entry-point example for the fitness-assistant RAG pipeline.

Demonstrates a keyword-search query through the full RAG pipeline.
"""

import json

from rag.pipeline import rag
from search.index import get_index

EXAMPLE_QUESTION = (
    "Is the Lat Pulldown considered a strength training activity, and if so, why?"
)


def main() -> None:
    index = get_index()
    result = rag(EXAMPLE_QUESTION, index)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
