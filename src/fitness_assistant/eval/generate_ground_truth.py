import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1]),
]

import pandas as pd
from tqdm.auto import tqdm

from llm.client import llm
from search.index import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "ground-truth-retrieval.csv"
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
CHECKPOINT_EVERY = 10

prompt_template = """
You are a fitness expert generating evaluation questions.

For the exercise below, generate 2 realistic questions that a gym user might ask.

The questions should be answerable using only the exercise information provided.

Return ONLY a JSON array of objects with the following format:

[
  {{
    "id": "<exercise_id>",
    "question": "<question>"
  }}
]

Exercise: {name}
Category: {category}
Body Part: {body_part}
Target Muscle: {target}
Primary Muscle Group: {muscle_group}
Secondary Muscles: {secondary_muscles}
Equipment: {equipment}

Instructions:
{instructions}
""".strip()


def build_prompt(doc: dict) -> str:
    
    return prompt_template.format(
        name=doc["name"],
        category=doc["category"],
        body_part=doc["body_part"],
        target=doc["target"],
        muscle_group=doc["muscle_group"],
        secondary_muscles=", ".join(doc["secondary_muscles"])
        if isinstance(doc["secondary_muscles"], list)
        else doc["secondary_muscles"],
        equipment=doc["equipment"],
        instructions=doc["instructions"]["en"]
        if isinstance(doc["instructions"], dict)
        else doc["instructions"],
    )


def extract_questions(response: str) -> list[dict]:
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("response is not a JSON array")
    questions = [
        q.get("question")
        for q in data
        if isinstance(q, dict) and isinstance(q.get("question"), str) and q["question"].strip()
    ]
    if not questions:
        raise ValueError("no valid questions in response")
    return questions[:3]


def generate_for_doc(doc: dict) -> list[dict]:
    prompt = build_prompt(doc)
    for attempt in range(MAX_RETRIES):
        try:
            questions = extract_questions(llm(prompt, model=MODEL))
            return [{"id": doc["id"], "question": q} for q in questions]
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise last_error


def sample_exercises(documents: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_part: dict[str, list[dict]] = {}
    for doc in documents:
        by_part.setdefault(doc["body_part"], []).append(doc)
    quota = {
        part: max(1, round(len(docs) * size / len(documents)))
        for part, docs in by_part.items()
    }
    sampled = []
    for part, docs in by_part.items():
        rng.shuffle(docs)
        sampled.extend(docs[: quota[part]])
    return sampled[:size]


def load_existing(output_path: Path) -> tuple[list[dict], set[str]]:
    if not output_path.exists():
        return [], set()
    df = pd.read_csv(output_path)
    if df.empty:
        return [], set()
    return df.to_dict(orient="records"), set(df["id"].astype(str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    documents = sample_exercises(load_documents(), args.sample_size, args.seed)
    existing, seen_ids = load_existing(args.output)
    rows = [dict(row) for row in existing]
    todo = [d for d in documents if str(d["id"]) not in seen_ids]
    if args.force:
        rows, todo = [], documents
    print(
        f"{len(documents)} sampled, {len(seen_ids)} already generated, "
        f"{len(todo)} to do"
    )

    for i, doc in enumerate(tqdm(todo), start=1):
        try:
            rows.extend(generate_for_doc(doc))
        except Exception as exc:
            print(f"skipping {doc['id']}: {exc}", file=sys.stderr)
            continue
        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} questions to {args.output}")


if __name__ == "__main__":
    main()
