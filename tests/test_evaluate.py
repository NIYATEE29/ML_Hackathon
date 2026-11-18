import numpy as np
import pandas as pd

from src.models.common import evaluate


def _toy():
    # one trip, 3 points; point 2's reference arc is missing from its candidates
    rows = [  # pt, cand_arc, from, to, label
        (0, 10, 0, 1, 1),
        (0, 11, 5, 6, 0),
        (1, 12, 1, 2, 1),
        (1, 13, 7, 8, 0),
        (2, 14, 2, 3, 0),
        (2, 15, 9, 9, 0),
    ]
    df = pd.DataFrame(rows, columns=["pt", "cand_arc", "from_node", "to_node", "label"])
    df["trip_id"], df["pt_has_label"] = "t", True
    return df


def test_perfect_scores_and_miss_rate():
    df = _toy()
    m = evaluate(df, df.label.to_numpy() * 0.9 + 0.05)
    assert abs(m["top1_acc"] - 2 / 3) < 1e-9  # point 2 is a candidate miss
    assert abs(m["candidate_miss_rate"] - 1 / 3) < 1e-9
    assert m["route_continuity"] == 1.0  # arc 10 (0->1) -> arc 12 (1->2) -> arc 14 (2->3)


def test_wrong_top1_lowers_accuracy():
    df = _toy()
    m = evaluate(df, np.where(df.label == 1, 0.1, 0.9))
    assert m["top1_acc"] == 0.0
