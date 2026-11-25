from src.context_budget import fit_context, answer_is_correct

def test_fit_context_keeps_rank_order_and_respects_budget():
    chunks = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
    selected = fit_context(chunks, budget=2)
    assert selected == chunks[:2]

def test_answer_quality_uses_answer_not_context():
    assert answer_is_correct("There are 12 casual leaves per year.", ["12 casual leaves"])
    assert not answer_is_correct("I don't have information on that", ["12 casual leaves"])
