from .config import DATA_DIR
from .llm import api_is_configured, call_llm, offline_extractive_answer

POLICY = (DATA_DIR / "company_leave_policy.txt").read_text(encoding="utf-8")
QUESTIONS = [
    ("How many casual leaves are allowed each year?", ["12 casual leaves"]),
    ("How many sick leaves are allowed each year?", ["10 sick leaves"]),
    ("How early should planned leave be requested?", ["3 working days"]),
    ("Who approves leave requests?", ["reporting manager"]),
    ("Can unused casual leave be carried forward?", ["cannot be carried forward"]),
]

def is_correct(answer, expected_phrases):
    text = answer.lower()
    return any(phrase.lower() in text for phrase in expected_phrases)

def run():
    print("TASK 1: WHY RAG - BASELINE VS GROUNDED")
    print("Policy source: data/company_leave_policy.txt")
    if not api_is_configured():
        print("\nNOTE: No API key is configured.")
        print("Baseline LLM calls are skipped honestly; no invented LLM response is shown.")
        print("To perform the required before/after LLM experiment, configure OPENAI_API_KEY in .env.")
    print("\n| Question | Baseline answer | Grounded answer |")
    print("|---|---|---|")
    for question, expected in QUESTIONS:
        if api_is_configured():
            try:
                baseline = call_llm(
                    "Answer this question using your existing knowledge. "
                    "Do not assume access to any private company policy. Question: " + question
                )
                baseline_status = "Correct" if is_correct(baseline, expected) else "Wrong/Invented"
            except Exception as exc:
                baseline = f"API error: {exc}"
                baseline_status = "Not evaluated"
        else:
            baseline = "NOT RUN - configure API key for a genuine LLM baseline"
            baseline_status = "Not evaluated"

        grounded_prompt = (
            "Answer ONLY using the context below. If the answer is absent, say "
            "'I don't have information on that'. Context: " + POLICY +
            " Question: " + question
        )
        if api_is_configured():
            try:
                grounded = call_llm(grounded_prompt)
            except Exception as exc:
                grounded = f"API error: {exc}"
        else:
            grounded = offline_extractive_answer(POLICY, question)

        grounded_status = "Correct" if is_correct(grounded, expected) else "Check answer"
        print(f"\nQ: {question}")
        print(f"Before context: {baseline}")
        print(f"Before status: {baseline_status}")
        print(f"After context: {grounded}")
        print(f"After status: {grounded_status}")

    print("\nWHY RAG OVER FINE-TUNING (5 LINES)")
    print("1. Company policies can change; update the source document and re-index it.")
    print("2. RAG retrieves current facts at question time.")
    print("3. Fine-tuning changes model weights and is not ideal for frequent factual updates.")
    print("4. Retrieved chunks can include source metadata for verification.")
    print("5. RAG avoids retraining the model for every policy change.")

if __name__ == "__main__":
    run()
