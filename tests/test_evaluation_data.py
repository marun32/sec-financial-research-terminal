import json
from collections import Counter
from pathlib import Path


def test_gold_set_has_required_coverage() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "gold_questions.jsonl"
    questions = [json.loads(line) for line in path.read_text().splitlines() if line]
    categories = Counter(item["category"] for item in questions)
    assert len(questions) == 30
    assert categories == {
        "numeric_table": 8,
        "narrative_risk": 8,
        "trend": 6,
        "cross_company": 4,
        "unanswerable": 4,
    }
    answerable = [item for item in questions if not item.get("unanswerable")]
    assert all(item.get("expected_tickers") for item in answerable)
    assert all(item.get("expected_forms") for item in answerable)
    assert all(item.get("expected_sections") for item in answerable)
    assert all(item.get("expected_terms") for item in answerable)
