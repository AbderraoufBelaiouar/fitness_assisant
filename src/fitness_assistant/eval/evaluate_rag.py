import json
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
from rag.pipeline import rag
from search.index import get_index
from load_ground_truth import load_ground_truth

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
VALID_RELEVANCE = {"NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"}

prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as 'NON_RELEVANT', 'PARTLY_RELEVANT', or 'RELEVANT'.

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def extract_evaluation(response: str) -> dict:
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"(?<=\W)'|'(?=\W)", '"', text)
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("evaluation is not a JSON object")
    relevance = data.get("Relevance")
    if relevance not in VALID_RELEVANCE:
        raise ValueError(f"invalid relevance: {relevance!r}")
    return data


def evaluate_rag(record: dict, index) -> tuple[dict, str, dict]:
    question = record["question"]
    answer_llm = rag(question, index, model=MODEL)

    prompt = prompt2_template.format(question=question, answer_llm=answer_llm)

    for attempt in range(MAX_RETRIES):
        try:
            evaluation = extract_evaluation(llm(prompt, model=MODEL))
            return record, answer_llm, evaluation
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise last_error


def main():
    sample = load_ground_truth()
    index = get_index()

    evaluations = []
    for record in tqdm(sample):
        try:
            evaluations.append(evaluate_rag(record, index))
        except Exception as exc:
            print(f"skipping {record['id']}: {exc}", file=sys.stderr)
            continue

    df_eval = pd.DataFrame(evaluations, columns=["record", "answer", "evaluation"])
    df_eval["id"] = df_eval.record.apply(lambda d: d["id"])
    df_eval["question"] = df_eval.record.apply(lambda d: d["question"])
    df_eval["relevance"] = df_eval.evaluation.apply(lambda d: d["Relevance"])
    df_eval["explanation"] = df_eval.evaluation.apply(lambda d: d["Explanation"])
    df_eval.to_csv("data/eval/rag-evaluations.csv", index=False)
    print(df_eval.relevance.value_counts(normalize=True))


if __name__ == "__main__":
    main()
