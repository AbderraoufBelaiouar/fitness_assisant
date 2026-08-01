import pandas as pd

from ingestion.documents import prepare_documents


def load_exercises():
    raw_documents = pd.read_json("data/exercises.json").to_dict(orient="records")
    docs = prepare_documents(raw_documents)
    pd.DataFrame(docs).to_json("data/processed/exercises_prepared.json", orient="records", indent=4)
    return docs
