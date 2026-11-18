"""The two route-continuity figures must measure different things.

`route_continuity` counts "still on the same arc" as continuous, which is why
it sits near 0.75 before a model does anything. `route_continuity_transitions`
only looks at consecutive points whose arc actually changed.
"""
import pandas as pd

from src.models.common import evaluate


def _trip():
    """Five points: stay, legal move, stay, illegal jump.

    arc 10 = node 0->1, arc 12 = node 1->2, arc 20 = node 7->8.
    So 10 -> 12 is a legal move (to_node 1 == from_node 1) and 12 -> 20 is not.
    Each point also carries a decoy candidate that never wins, so both classes
    are present for the row-level metrics.
    """
    real = {0: (10, 0, 1), 1: (10, 0, 1), 2: (12, 1, 2), 3: (12, 1, 2), 4: (20, 7, 8)}
    rows = []
    for pt, (arc, u, v) in real.items():
        rows.append((pt, arc, u, v, 1, 0.9))
        rows.append((pt, 99, 5, 6, 0, 0.1))   # decoy
    df = pd.DataFrame(rows, columns=["pt", "cand_arc", "from_node", "to_node", "label", "p"])
    df["trip_id"], df["pt_has_label"] = "t", True
    return df


def test_plain_continuity_counts_staying_on_the_same_arc():
    df = _trip()
    m = evaluate(df, df.p.to_numpy())
    # pairs: (same, legal, same, illegal) -> 3 of 4 count as continuous
    assert abs(m["route_continuity"] - 0.75) < 1e-9


def test_transition_continuity_ignores_staying_put():
    df = _trip()
    m = evaluate(df, df.p.to_numpy())
    # only two pairs actually change arc, and one of them is illegal
    assert m["n_arc_transitions"] == 2
    assert abs(m["route_continuity_transitions"] - 0.5) < 1e-9


def test_transition_continuity_is_the_stricter_of_the_two():
    df = _trip()
    m = evaluate(df, df.p.to_numpy())
    assert m["route_continuity_transitions"] < m["route_continuity"]


def test_a_route_that_never_changes_arc_has_no_transitions():
    df = _trip()
    df.loc[df.label == 1, ["cand_arc", "from_node", "to_node"]] = [10, 0, 1]
    m = evaluate(df, df.p.to_numpy())
    assert m["route_continuity"] == 1.0
    assert m["n_arc_transitions"] == 0
    assert pd.isna(m["route_continuity_transitions"])
