import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1]),
]

import pandas as pd

from metrics import evaluate
from search.index import get_index
from search.search import search

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "eval" / "ground-truth-retrieval.csv"


def load_ground_truth(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--boost", type=json.loads, default={})
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    index = get_index()

    def search_fn(q: dict) -> list[dict]:
        return search(
            q["question"], index, boost=args.boost, num_results=args.num_results
        )

    metrics = evaluate(ground_truth, search_fn)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
