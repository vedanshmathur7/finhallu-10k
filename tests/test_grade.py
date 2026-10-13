from finhallu.grade import extract_number, grade
from finhallu.questions import build

REV = {"unit": "usd_millions", "answer": 416161, "prior_answer": 391035}


def test_extract_scales():
    assert extract_number("FINAL ANSWER: $416.2 billion", "usd_millions") == 416200
    assert extract_number("Revenue was $416,161 million in fiscal 2025.", "usd_millions") == 416161
    assert extract_number("FINAL ANSWER: 416,161", "usd_millions") == 416161
    assert extract_number("$416,161,000,000", "usd_millions") == 416161
    assert extract_number("FINAL ANSWER: $(147,782) million", "usd_millions") == -147782


def test_percent_direction():
    assert extract_number("FINAL ANSWER: decreased 4.96%", "percent") == -4.96
    assert extract_number("FINAL ANSWER: 6.43%", "percent") == 6.43


def test_labels():
    assert grade(REV, "FINAL ANSWER: $416.2 billion") == "correct"
    assert grade(REV, "FINAL ANSWER: $391,035 million") == "wrong_year"
    assert grade(REV, "FINAL ANSWER: $416,161 thousand") == "scale_error"
    assert grade(REV, "FINAL ANSWER: $500,000 million") == "wrong_value"
    assert grade(REV, "FINAL ANSWER: NOT_IN_DOCUMENT") == "false_refusal"
    assert grade(REV, "I'm not sure.") == "no_answer"
    neg = {"unit": "usd_millions", "answer": -147782, "prior_answer": -42012}
    assert grade(neg, "FINAL ANSWER: $147,782 million") == "sign_error"


def test_traps():
    trap = {"unit": "abstain", "answer": None}
    assert grade(trap, "FINAL ANSWER: NOT_IN_DOCUMENT") == "correct"
    assert grade(trap, "The filing covers fiscal 2025 only; fiscal 2028 revenue is not provided.") == "correct"
    assert grade(trap, "FINAL ANSWER: $512 billion") == "fabricated"


def test_question_set_shape():
    qs = build()
    assert len(qs) >= 150
    assert len({q["id"] for q in qs}) == len(qs)
    assert {q["type"] for q in qs} == {"lookup", "prior_year", "yoy", "ratio", "trap", "multiturn"}
