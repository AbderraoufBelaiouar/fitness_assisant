def search(query, index):
    boost = {
        "name": 1.785508945415859,
        "category": 1.2214864078353629,
        "body_part": 1.5007801416431321,
        "equipment": 2.7187580803181715,
        "muscle_group": 0.4691445588748365,
        "target": 1.348246061991619,
        "secondary_muscles": 0.9007706510043623,
        "instructions": 1.9338680512343536,
    }

    results = index.search(
        query=query, filter_dict={}, boost_dict=boost, num_results=10
    )

    return results

