import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

from search.embedder import embed_documents
from search.index import load_documents

OUTPUT = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "exercises_embeddings.json"
)


def main():
    chunks, vectors = embed_documents(load_documents())
    payload = [
        {"chunk": chunk, "vector": vector.tolist()}
        for chunk, vector in zip(chunks, vectors)
    ]
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"wrote {len(payload)} chunk embeddings to {OUTPUT}")


if __name__ == "__main__":
    main()
