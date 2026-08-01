from rag.pipeline import rag
from search.index import get_index, get_vector_index
from search.search import search
from search.vector import vector_search

index = get_index()
vindex = get_vector_index()
question = (
    "Is the Lat Pulldown considered a strength training activity, and if so, why?"
)

vector_results = vector_search(question, vindex)
search_results = search(question, index)

# answer = rag(question, index)
# print(answer)
