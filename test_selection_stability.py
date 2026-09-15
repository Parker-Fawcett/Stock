import pandas as pd

from selection_stability import selection


def test_selection_applies_threshold_and_top_count_boundary():
    frame = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "proba_legacy": [0.40, 0.35, 0.30, 0.20],
    })

    picks, boundary = selection(frame, "legacy", threshold=0.25, top_n=2)

    assert picks == {"A", "B"}
    assert boundary == 0.35


def test_selection_uses_probability_threshold_when_book_is_not_full():
    frame = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "proba_corrected": [0.40, 0.24, 0.20],
    })

    picks, boundary = selection(frame, "corrected", threshold=0.25, top_n=2)

    assert picks == {"A"}
    assert boundary == 0.25
