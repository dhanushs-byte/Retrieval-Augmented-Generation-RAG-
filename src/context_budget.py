import tiktoken
from .embeddings import embed_query
from .vector_store import search_chunks
from .llm import api_is_configured, call_llm, offline_extractive_answer
from .config import SIMILARITY_THRESHOLD
from .query_pipeline import PROMPT_TEMPLATE

ENCODING = tiktoken.get_encoding("cl100k_base")
SYSTEM_PROMPT = "You are a careful assistant. Use only the supplied context."
DEFAULT_RESERVED_OUTPUT = 500

def count_tokens(text):
    return len(ENCODING.encode(text or ""))

def fit_context(chunks, budget):
    selected, used = [], 0
    for chunk in chunks:
        cost = count_tokens(chunk["text"])
        if used + cost > budget:
            break
        selected.append(chunk)
        used += cost
    return selected

def calculate_context_budget(question, chunks, context_window=4000,
                             reserved_output=DEFAULT_RESERVED_OUTPUT):
    question_tokens = count_tokens(question)
    system_tokens = count_tokens(SYSTEM_PROMPT)
    fixed_prompt_tokens = count_tokens("Answer ONLY using the context below. Context:  Question: ")
    available = max(0, context_window - system_tokens - question_tokens
                    - fixed_prompt_tokens - reserved_output)
    selected = fit_context(chunks, available)
    context = "\\n\\n".join(c["text"] for c in selected)
    actual_prompt_tokens = system_tokens + count_tokens(
        PROMPT_TEMPLATE.format(context=context, question=question)
    ) + reserved_output
    return selected, {
        "system_prompt_tokens": system_tokens,
        "retrieved_chunk_tokens": sum(count_tokens(c["text"]) for c in selected),
        "question_tokens": question_tokens,
        "reserved_output_tokens": reserved_output,
        "estimated_total_tokens": actual_prompt_tokens,
        "context_window": context_window
    }

def generate_context_answer(question, chunks):
    context = "\\n\\n".join(c["text"] for c in chunks)
    if not context:
        return "I don't have information on that"
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    if api_is_configured():
        return call_llm(prompt)
    return offline_extractive_answer(context, question)

def answer_is_correct(answer, expected_phrases):
    answer = answer.lower()
    return any(phrase.lower() in answer for phrase in expected_phrases)

def run_budget_experiment():
    question = "How many casual leaves per year?"
    expected = ["12 casual leaves", "12 casual leave"]
    retrieved = search_chunks(embed_query(question), limit=10)
    print("k | context tokens | answer correct (Y/N) | answer")
    for k in (2, 5, 10):
        chosen, stats = calculate_context_budget(question, retrieved[:k], 4000)
        generated = generate_context_answer(question, chosen)
        correct = answer_is_correct(generated, expected)
        print(f"{k} | {stats['retrieved_chunk_tokens']} | "
              f"{'Y' if correct else 'N'} | {generated}")
        print("  Token breakdown:", stats)

    print("\\n8K CONTEXT MODEL - RECOMMENDED STARTING SPLIT")
    print("System/instructions: 800 tokens (10%)")
    print("Retrieved context: 4,000 tokens (50%)")
    print("Question/history: 1,200 tokens (15%)")
    print("Reserved output: 2,000 tokens (25%)")
    print("These are starting allocations; measure real prompts and adjust.")

def compare_chunk_positions(question, chunks):
    if len(chunks) < 2:
        print("Need at least two chunks to compare ordering.")
        return
    best = chunks[0]
    remaining = chunks[1:]
    orders = {
        "Best chunk first": [best] + remaining,
        "Best chunk last": remaining + [best],
    }
    print("\\nLOST-IN-THE-MIDDLE ORDER COMPARISON")
    for label, ordered in orders.items():
        result = generate_context_answer(question, ordered)
        print(f"\\n{label}:")
        print("Answer:", result)
    print("\\nCompare both answers against the source document. The effect is model- and")
    print("prompt-dependent; this small experiment does not guarantee one order wins.")

if __name__ == "__main__":
    run_budget_experiment()
    q = "How many casual leaves per year?"
    chunks = search_chunks(embed_query(q), limit=10)
    compare_chunk_positions(q, chunks)
