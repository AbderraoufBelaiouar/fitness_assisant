from tqdm.auto import tqdm


def hit_rate(relevance_total):
    if not relevance_total:
        return 0.0
    hits = sum(1 for line in relevance_total if True in line)
    return hits / len(relevance_total)


def mrr(relevance_total):
    if not relevance_total:
        return 0.0
    total_score = 0.0
    for line in relevance_total:
        for rank, relevant in enumerate(line):
            if relevant:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q["id"]
        results = search_function(q)
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }
