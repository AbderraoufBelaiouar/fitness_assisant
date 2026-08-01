
from llm.client import llm
from llm.prompts import build_prompt
from search.search import search


def rag(query, index, model="llama-3.3-70b-versatile"):
    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer