from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "eval" / "ground-truth-retrieval.csv"


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records")
