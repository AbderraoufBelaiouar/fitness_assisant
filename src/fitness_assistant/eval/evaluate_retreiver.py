import argparse
import json
import sys
from pathlib import Path


sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1]),
]

import pandas as pd
from search.vector import vector_search

from metrics import evaluate
from search.index import get_index, get_vector_index
from search.search import search
from load_ground_truth import load_ground_truth
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "eval" / "ground-truth-retrieval.csv"




def main():

    ground_truth = load_ground_truth(DEFAULT_GROUND_TRUTH)
    index = get_index()
    vindex=get_vector_index()

    def search_fn(q: dict) -> list[dict]:
        return search(
            q["question"], index
        )
    def vector_search_fn(q: dict) -> list[dict]:
        return vector_search(
            q["question"], vindex
        )

    keyword_metrics = evaluate(ground_truth, search_fn)
    vector_metrics = evaluate(ground_truth, vector_search_fn)
    print(json.dumps(keyword_metrics, indent=2))
    print(json.dumps(vector_metrics, indent=2))


if __name__ == "__main__":
    main()
