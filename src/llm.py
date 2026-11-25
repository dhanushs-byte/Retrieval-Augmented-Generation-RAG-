from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

def api_is_configured():
    return bool(OPENAI_API_KEY)

def call_llm(prompt):
    """Call the configured OpenAI-compatible endpoint; no silent fake LLM."""
    if not api_is_configured():
        raise RuntimeError(
            "No LLM API key configured. Add OPENAI_API_KEY to .env. "
            "Offline mode can answer from retrieved context, but cannot demonstrate "
            "an LLM's baseline hallucination."
        )
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def offline_extractive_answer(context, question):
    """Simple transparent fallback: choose the context sentence with most query-term overlap."""
    import re
    if not context.strip():
        return "I don't have information on that"
    terms = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 2}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    if not sentences:
        return "I don't have information on that"
    ranked = sorted(
        sentences,
        key=lambda s: sum(term in s.lower() for term in terms),
        reverse=True
    )
    if not terms or sum(term in ranked[0].lower() for term in terms) == 0:
        return "I don't have information on that"
    return ranked[0]
