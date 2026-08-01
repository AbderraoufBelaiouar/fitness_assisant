import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1]),
]

import pandas as pd
from tqdm.auto import tqdm

from llm.client import llm
from metrics import evaluate
from search.index import get_index, load_documents
from search.search import search

prompt1_template = """
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


def generate_ground_truth_questions():
    results = []
    df = pd.DataFrame(load_documents())
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = prompt1_template.format(
            name=row["name"],
            category=row["category"],
            body_part=row["body_part"],
            target=row["target"],
            muscle_group=row["muscle_group"],
            secondary_muscles=", ".join(row["secondary_muscles"])
            if isinstance(row["secondary_muscles"], list)
            else row["secondary_muscles"],
            equipment=row["equipment"],
            instructions=row["instructions"]["en"]
            if isinstance(row["instructions"], dict)
            else row["instructions"],
        )

        response = llm(prompt, model="llama-3.3-70b-versatile")

        questions = json.loads(response)

        for q in questions:
            q["id"] = row["id"]
            results.append(q)

    df_questions = pd.DataFrame(results)
    df_questions.to_csv("data/eval/ground-truth-retrieval.csv", index=False)


def load_ground_truth_questions():
    df = pd.read_csv("data/eval/ground-truth-retrieval.csv")
    return df.to_dict(orient="records")


if __name__ == "__main__":
    # generate_ground_truth_questions()
    ground_truth = load_ground_truth_questions()
    index = get_index()
    print(evaluate(ground_truth, lambda q: search(q["question"], index)))
