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


# Regression cases taken from the first real run (Phi-3.5-mini on Kaggle)
def test_form_name_is_not_a_number():
    q = {"unit": "usd_millions", "answer": 112010, "prior_answer": 93736}
    a = "The provided 10-K excerpts do not contain the specific figure for Apple Inc.'s net income for fiscal 2025."
    assert grade(q, a) == "false_refusal"


def test_date_is_not_a_fabrication():
    trap = {"unit": "abstain", "answer": None}
    a = "According to the excerpts from Apple's 10-K for the fiscal year ended September 27, 2025, fiscal 2028 revenue is not provided."
    assert grade(trap, a) == "correct"
    assert grade(trap, "For the fiscal year ended September 27, revenue will be $512 billion.") == "fabricated"


def test_precision_aware_tolerance():
    q = {"unit": "usd_millions", "answer": 130387, "prior_answer": 81453}
    assert grade(q, "130,141 million dollars") == "wrong_value"  # stated to the million, off by 246
    assert grade(q, "$130.4 billion") == "correct"                # stated to 0.1bn, within 50m
    assert grade({"unit": "usd_millions", "answer": 182935, "prior_answer": 1}, "$182.9 billion") == "correct"


def test_real_answer_shapes():
    assert grade(REV, "Apple Inc.'s total revenue (net sales) for fiscal 2025 was $416.161 million.") == "scale_error"
    assert grade({"unit": "percent", "answer": 6.43}, "FINAL ANSWER: 6%") == "correct"
    assert grade({"unit": "usd_per_share", "answer": 4.9, "prior_answer": 2.94}, "4.90") == "correct"
    assert grade({"unit": "usd_millions", "answer": 111482, "prior_answer": 1}, "111,482 million dollars") == "correct"
    assert grade(REV, "FINAL ANSWER: NOT_IN_DOCUMENT") == "false_refusal"


# Regression cases from run 4 (Qwen/Granite on Kaggle)
def test_percent_ignores_dollar_working():
    q = {"unit": "percent", "answer": 60.08}
    a = "We use the values:\n- Operating income fiscal 2026: $130,387 million\n- fiscal 2025: $81,453 million"
    assert grade(q, a) == "no_answer"  # reasoning cut off before the %: not a wrong value
    assert grade(q, a + "\nChange = 60.1%\nFINAL ANSWER: 60.1%") == "correct"


def test_infra_error_excluded():
    assert grade(REV, "[ERROR] CUDA out of memory") == "infra_error"
