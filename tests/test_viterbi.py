import numpy as np
import pandas as pd

from src.hmm.viterbi import ArcGraph, HMMParams, decode_trip


def _graph():
    # arcs: 0:(0->1) 1:(1->2) 2:(2->3) 3:(7->8) ; arc 3 is far from everything
    arcs = np.array([[0, 1], [1, 2], [2, 3], [7, 8]])
    return arcs, ArcGraph(arcs, 9)


def _rows(spec, dt=1.0):
    # spec: list per point of [(arc, p)]
    arcs, _ = _graph()
    out = []
    for pt, cs in enumerate(spec):
        for a, p in cs:
            out.append(dict(pt=pt, cand_arc=a, from_node=arcs[a, 0], to_node=arcs[a, 1], p=p, log_dt_prev=np.log1p(dt)))
    return pd.DataFrame(out)


def test_connectivity_overrides_a_noisy_classifier():
    _, g = _graph()
    # the middle point's classifier prefers the disconnected arc 3 (0.6) over arc 1 (0.4)
    df = _rows([[(0, 0.9)], [(3, 0.6), (1, 0.4)], [(2, 0.9)]])
    assert decode_trip(df, g, HMMParams()).cand_arc.tolist() == [0, 1, 2]
    assert df.loc[df.groupby("pt").p.idxmax()].cand_arc.tolist() == [0, 3, 2]   # classifier-only is wrong


def test_point_without_candidates_is_skipped_and_gap_reset_works():
    _, g = _graph()
    df = _rows([[(0, 0.9)], [(1, 0.9)]])
    df.loc[df.pt == 1, "pt"] = 5                 # points 1..4 have no candidates
    out = decode_trip(df, g, HMMParams())
    assert out.pt.tolist() == [0, 5]
    # a huge time gap resets the chain: decode follows the classifier even across a break
    df2 = _rows([[(0, 0.9)], [(3, 0.9)]], dt=500.0)
    assert decode_trip(df2, g, HMMParams()).cand_arc.tolist() == [0, 3]
