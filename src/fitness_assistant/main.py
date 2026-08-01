from rag.pipeline import rag
from search.index import get_index

index = get_index()
question = (
    "Is the Lat Pulldown considered a strength training activity, and if so, why?"
)
answer = rag(question, index)
print(answer)
