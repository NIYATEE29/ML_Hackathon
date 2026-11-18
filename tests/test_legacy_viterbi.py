import pytest
import numpy as np
from src.hmm.legacy_viterbi import PathSnapHMMDecoder

def test_viterbi_synthetic():
    # Simple linear graph: Node 0 -> Node 1 -> Node 2
    # Arc 0: 0 -> 1
    # Arc 1: 1 -> 2
    nodes = np.array([
        [-1.8400, 52.9500],
        [-1.8400, 52.9510],
        [-1.8400, 52.9520]
    ])
    arcs = np.array([
        [0, 1],
        [1, 2]
    ])

    decoder = PathSnapHMMDecoder(nodes, arcs)

    # 2 observations, each with candidate arcs [0, 1]
    candidate_lists = [[0, 1], [0, 1]]
    candidate_probs = [[0.9, 0.1], [0.1, 0.9]]
    candidate_dists = [[2.0, 20.0], [20.0, 2.0]]
    gps_xy = np.array([[0.0, 0.0], [0.0, 100.0]])
    time_deltas = np.array([0.0, 1.0])

    route = decoder.decode(
        candidate_lists=candidate_lists,
        candidate_probs=candidate_probs,
        candidate_dists=candidate_dists,
        gps_xy=gps_xy,
        time_deltas=time_deltas
    )

    assert route == [0, 1]
