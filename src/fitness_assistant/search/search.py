def search(query, index, boost=None, num_results=10):
    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost or {},
        num_results=num_results,
    )
    return results
