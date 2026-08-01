from ingestion.chunking import chunk_documents

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_documents(documents: list[dict], chunk_size: int = 200, batch_size: int = 50):
    chunks = chunk_documents(documents, chunk_size)
    texts = [chunk["instructions"] for chunk in chunks]
    vectors = get_model().encode(texts, batch_size=batch_size)
    return chunks, vectors
