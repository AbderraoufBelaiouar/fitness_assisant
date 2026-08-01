def chunk_documents(documents: list[dict], chunk_size: int = 200) -> list[dict]:
    """
    Chunk the documents into smaller pieces based on the specified chunk size.

    Args:
        documents (list[dict]): List of documents to be chunked.
        chunk_size (int): The maximum number of words in each chunk.

    Returns:
        list[dict]: List of chunked documents.
    """
    chunked_documents = []
    for doc in documents:
        text = doc.get("instructions", "")
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                new_doc = doc.copy()
                new_doc["instructions"] = chunk
                chunked_documents.append(new_doc)
    return chunked_documents