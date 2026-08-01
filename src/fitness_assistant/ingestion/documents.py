def prepare_document(doc: dict) -> dict:
    return {
        **doc,
        "instructions": doc["instructions"]["en"],
        "instruction_steps": " ".join(doc["instruction_steps"]["en"]),
        "secondary_muscles": " ".join(doc["secondary_muscles"]),
    }


def prepare_documents(documents: list[dict]) -> list[dict]:
    return [prepare_document(doc) for doc in documents]
