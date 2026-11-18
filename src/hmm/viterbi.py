"""HMM / Viterbi route reconstruction on top of classifier candidate scores.

Hidden states
    At every GPS point t the state is one of its candidate arcs (<= 12 per point, from
    src.features.candidates). The candidate -> road-arc correspondence is kept, so a decoded
    state IS a road segment id.

Emission  (per point, per candidate)
    emission_weight * log(max(p, p_floor)), where p is the classifier's P(candidate is the
    travelled arc). Any per-point constant would cancel in Viterbi, so raw p is enough.

Transition  (previous point's arc a  ->  this point's arc b)  -- log-scores
    * same arc                                   : 0
    * b starts where a ends (0 arcs in between)  : -(switch_penalty)
    * reachable with h intermediate arcs (BFS on the directed arc graph, h <= H_eff)
                                                 : -(switch_penalty + hop_penalty * h)
    * immediate U-turn (b is the reverse of a)   : -reverse_penalty
    * not reachable within H_eff hops            : -disconnected_penalty   (a finite floor, so
                                                   decoding continues across broken links
                                                   instead of failing)
    H_eff = max_hops + hops_per_sec * (dt - 1), capped at BFS_DEPTH: a longer time gap lets the
    vehicle cover more arcs.  If dt > gap_reset_s the transition term is dropped (all zeros),
    i.e. the chain restarts from the classifier alone.

Missing candidates
    A point with no candidate gets no decoded arc (-1); the chain bridges over it using the
    elapsed time between the neighbouring decoded points.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BFS_DEPTH = 10


@dataclass
class HMMParams:
    emission_weight: float = 1.0
    p_floor: float = 1e-4
    switch_penalty: float = 0.5
    hop_penalty: float = 0.5
    reverse_penalty: float = 3.0
    disconnected_penalty: float = 6.0
    max_hops: int = 2
    hops_per_sec: float = 0.5
    gap_reset_s: float = 30.0

    def as_dict(self):
        return asdict(self)


class ArcGraph:
    """Directed node graph of one trip's arcs with memoised bounded BFS."""

    def __init__(self, arcs: np.ndarray, n_nodes: int):
        order = np.argsort(arcs[:, 0], kind="stable")
        self.dst = arcs[order, 1]
        self.ptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(self.ptr, arcs[:, 0] + 1, 1)
        self.ptr = np.cumsum(self.ptr)
        self._memo: dict[int, dict[int, int]] = {}

    def hops_from(self, node: int) -> dict[int, int]:
        """node -> number of arcs on the shortest directed path from `node`, up to BFS_DEPTH."""
        hit = self._memo.get(node)
        if hit is not None:
            return hit
        dist, frontier = {node: 0}, [node]
        for d in range(1, BFS_DEPTH + 1):
            nxt = []
            for u in frontier:
                for v in self.dst[self.ptr[u] : self.ptr[u + 1]]:
                    v = int(v)
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
        self._memo[node] = dist
        return dist


def _transition(prev_arc, prev_from, prev_to, cur_arc, cur_from, cur_to, graph, h_eff, P: HMMParams):
    """Log-score matrix [len(prev), len(cur)]."""
    T = np.empty((len(prev_arc), len(cur_arc)))
    for i in range(len(prev_arc)):
        reach = graph.hops_from(int(prev_to[i]))
        for j in range(len(cur_arc)):
            if prev_arc[i] == cur_arc[j]:
                T[i, j] = 0.0
            elif cur_from[j] == prev_to[i] and cur_to[j] == prev_from[i]:
                T[i, j] = -P.reverse_penalty
            else:
                h = reach.get(int(cur_from[j]))
                T[i, j] = (
                    -(P.switch_penalty + P.hop_penalty * h)
                    if (h is not None and h <= h_eff)
                    else -P.disconnected_penalty
                )
    return T


def decode_trip(cands: pd.DataFrame, graph: ArcGraph, params: HMMParams | None = None) -> pd.DataFrame:
    """Viterbi decode one trip.

    cands: candidate rows of ONE trip with columns pt, cand_arc, from_node, to_node, p, log_dt_prev.
    Returns one row per decoded point: pt, cand_arc, from_node, to_node (points without
    candidates are absent).
    """
    P = params or HMMParams()
    cands = cands.sort_values(["pt", "cand_arc"], kind="stable")
    pts = cands["pt"].to_numpy()
    starts = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1]])
    ends = np.r_[starts[1:], len(pts)]
    arc, fr, to = (cands[c].to_numpy() for c in ("cand_arc", "from_node", "to_node"))
    emis = P.emission_weight * np.log(np.maximum(cands["p"].to_numpy(), P.p_floor))
    dt_prev = np.expm1(cands["log_dt_prev"].to_numpy())

    n = len(starts)
    delta = [emis[starts[0] : ends[0]]]
    back = []
    for k in range(1, n):
        s0, e0, s1, e1 = starts[k - 1], ends[k - 1], starts[k], ends[k]
        skipped = int(pts[s1] - pts[s0])
        dt = dt_prev[s1] * skipped
        if dt > P.gap_reset_s:
            T = np.zeros((e0 - s0, e1 - s1))
        else:
            h_eff = min(BFS_DEPTH, P.max_hops + int(P.hops_per_sec * max(0.0, dt - 1.0)))
            T = _transition(arc[s0:e0], fr[s0:e0], to[s0:e0], arc[s1:e1], fr[s1:e1], to[s1:e1], graph, h_eff, P)
        score = delta[-1][:, None] + T
        bp = score.argmax(axis=0)
        delta.append(score[bp, np.arange(e1 - s1)] + emis[s1:e1])
        back.append(bp)

    choice = np.empty(n, dtype=int)
    choice[-1] = int(delta[-1].argmax())
    for k in range(n - 1, 0, -1):
        choice[k - 1] = back[k - 1][choice[k]]
    rows = starts + choice
    return pd.DataFrame({"pt": pts[rows], "cand_arc": arc[rows], "from_node": fr[rows], "to_node": to[rows]})
