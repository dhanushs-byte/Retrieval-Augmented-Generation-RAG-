from .embeddings import embed_query
from .vector_store import search_chunks
from .llm import call_llm, api_is_configured, offline_extractive_answer
from .config import SIMILARITY_THRESHOLD, TOP_K

PROMPT_TEMPLATE = "Answer ONLY using the context below. Context: {context} Question: {question}"

def answer(question, k=4, threshold=None, debug=True, chunks_override=None):
    threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
    retrieved = chunks_override if chunks_override is not None else search_chunks(embed_query(question), limit=k)
    if debug:
        print(f"\\nQUESTION: {question}")
        print("RETRIEVED CHUNKS:")
        for i, chunk in enumerate(retrieved, 1):
            print(f"  [{i}] score={chunk.get('score', 0):.4f} source={chunk.get('source')} "
                  f"page={chunk.get('page')} section={chunk.get('section')}")
            print(f"      {chunk['text']}")
    if not retrieved or retrieved[0].get("score", 0) < threshold:
        result = "I don't have information on that"
        if debug:
            print("ANSWER:", result, "(fallback; LLM not called)")
        return result
    context = "\\n\\n".join(c["text"] for c in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    if api_is_configured():
        result = call_llm(prompt)
    else:
        result = offline_extractive_answer(context, question)
    if debug:
        print("ANSWER:", result)
    return result

if __name__ == "__main__":
    questions = [
        "How many casual leaves per year?",
        "How many sick leaves are provided?",
        "How many days in advance should leave be requested?",
        "Who approves a leave request?",
        "Can unused casual leave be carried forward?",
        "What is the process for emergency leave?",
        "Who won the world cup?",
        "What is the capital of Mars?",
        "What is the company's office located on the Moon?",
        "How many elephants work in the finance department?",
    ]
    for q in questions:
        answer(q, k=TOP_K)
