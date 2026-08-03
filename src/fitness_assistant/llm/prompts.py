"""Prompt templates and builders for the fitness-assistant RAG pipeline."""

from typing import Union

_SYSTEM_PROMPT = """
You're a fitness instructor. Answer the QUESTION using only the information provided in the CONTEXT from the exercise database.

If the answer cannot be found in the CONTEXT, say that you don't have enough information.

QUESTION:
{question}

CONTEXT:
{context}
""".strip()

_ENTRY_TEMPLATE = """
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


def _format_secondary_muscles(value: Union[list, str]) -> str:
    """Normalise secondary_muscles regardless of whether it is a list or a
    pre-joined string (the processed dataset stores it as a string)."""
    if isinstance(value, list):
        return ", ".join(value)
    return value


def build_prompt(query: str, search_results: list[dict]) -> str:
    """Construct a RAG prompt from retrieved exercise documents.

    Args:
        query: The user's question.
        search_results: A list of exercise dicts returned by the retriever.

    Returns:
        A fully-formatted prompt string ready to be sent to the LLM.
    """
    entries = []
    for doc in search_results:
        entries.append(
            _ENTRY_TEMPLATE.format(
                name=doc.get("name", ""),
                category=doc.get("category", ""),
                body_part=doc.get("body_part", ""),
                target=doc.get("target", ""),
                muscle_group=doc.get("muscle_group", ""),
                secondary_muscles=_format_secondary_muscles(
                    doc.get("secondary_muscles", "")
                ),
                equipment=doc.get("equipment", ""),
                instructions=doc.get("instructions", ""),
            )
        )

    context = "\n\n".join(entries)
    return _SYSTEM_PROMPT.format(question=query, context=context)