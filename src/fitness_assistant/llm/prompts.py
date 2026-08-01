prompt_template = """
You're a fitness instructor. Answer the QUESTION using only the information provided in the CONTEXT from the exercise database.

If the answer cannot be found in the CONTEXT, say that you don't have enough information.

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


entry_template = """
Exercise: {name}
Category: {category}
Body Part: {body_part}
Target Muscle: {target}
Primary Muscle Group: {muscle_group}
Secondary Muscles: {secondary_muscles}
Equipment: {equipment}

Instructions:
{instructions}
""".strip()


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context += entry_template.format(
            name=doc["name"],
            category=doc["category"],
            body_part=doc["body_part"],
            target=doc["target"],
            muscle_group=doc["muscle_group"],
            secondary_muscles=", ".join(doc["secondary_muscles"]),
            equipment=doc["equipment"],
            instructions=doc["instructions"],
        )
        context += "\n\n"

    return prompt_template.format(
        question=query,
        context=context.strip(),
    )