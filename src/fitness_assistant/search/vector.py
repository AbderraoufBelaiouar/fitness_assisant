from search.embedder import get_model


def vector_search(query, vindex, num_results=10):
    results = vindex.search(
        get_model().encode(query), num_results=num_results
    )
    return results
