"""Candidate generation, order-aware labelling and feature engineering.

Problem framing: candidate scoring. For every GPS point we generate the nearby road
arcs (candidates). Each (point, candidate) row gets label 1 if the candidate is the
reference route arc for that point. A model scores each row; the arg-max candidate per
point is the Top-1 prediction and the scores feed the HMM later.

Everything is computed in per-trip projected metres (see src/data/loader.py).
Node/arc ids are local to a trip, so the table is always grouped by `trip_id`.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.loader import Trip

CAND_RADIUS_M = 40.0  # candidate search radius
MAX_CANDS = 12  # keep at most this many nearest candidates per point
DENSIFY_STEP_M = 10.0  # arc sampling step for the KD-tree
LABEL_MAX_DIST_M = 30.0  # a point farther than this from the route gets no label
LABEL_WINDOW = 60  # forward window (in route arcs) for order-aware labelling
LABEL_BACK = 3  # allowed backward slack (in route arcs)
LABEL_RESET_AFTER = 20  # unmatched points before the route pointer is re-anchored
CTX_OFFSETS = (-5, -3, -1, 1, 3, 5)  # neighbour offsets for context features

FEATURES = [
    "dist",
    "dist_rank",
    "dist_minus_min",
    "n_cands",
    "t_frac",
    "along_m",
    "arc_len",
    "cos_h1",
    "cos_h3",
    "speed",
    "log_dt_prev",
    "log_dt_next",
    "has_reverse",
    "conn_prev",
    "rev_prev",
    "conn_next",
    "rev_next",
    "d_m5",
    "d_m3",
    "d_m1",
    "d_p1",
    "d_p3",
    "d_p5",
    "d_ctx_mean",
    "d_ctx_max",
    "deg_from",
    "deg_to",
]


def _densify(N: np.ndarray, arcs: np.ndarray, step: float):
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    L = np.hypot(*(B - A).T)
    k = np.maximum(2, np.ceil(L / step).astype(int) + 1)
    arc_idx = np.repeat(np.arange(len(arcs)), k)
    start = np.cumsum(k) - k
    j = np.arange(k.sum()) - np.repeat(start, k)
    t = j / np.repeat(k - 1, k)
    pts = A[arc_idx] + t[:, None] * (B - A)[arc_idx]
    return pts, arc_idx


def _seg_dist(P, A, B):
    """Distance from points P to segments AB, plus clipped projection fraction."""
    AB = B - A
    L2 = np.maximum((AB**2).sum(1), 1e-9)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0.0, 1.0)
    d = np.hypot(*(P - (A + t[:, None] * AB)).T)
    return d, t


def label_points(P: np.ndarray, N: np.ndarray, arcs: np.ndarray, route: np.ndarray):
    """Order-aware pseudo-label: the route arc each GPS point is travelling on.

    Returns (true_arc[int, -1 if unlabelled], dist_to_true_arc[float]).
    The route pointer only moves forward (small backward slack), which prevents a point
    near a revisited or parallel stretch from snapping to the wrong occurrence of an arc.
    """
    ra = arcs[route]
    pts, ridx = _densify(N, ra, 5.0)
    tree = cKDTree(pts)
    lists = tree.query_ball_point(P, LABEL_MAX_DIST_M + 3.0)
    true_arc = np.full(len(P), -1, dtype=int)
    true_d = np.full(len(P), np.nan)
    ptr, miss_run = 0, 0
    for i, l in enumerate(lists):
        if len(l) == 0:
            miss_run += 1
            continue
        pos = np.unique(ridx[l])
        d, _ = _seg_dist(np.repeat(P[i][None], len(pos), 0), N[ra[pos, 0]], N[ra[pos, 1]])
        ok = d <= LABEL_MAX_DIST_M
        if not ok.any():
            miss_run += 1
            continue
        pos, d = pos[ok], d[ok]
        win = (pos >= ptr - LABEL_BACK) & (pos <= ptr + LABEL_WINDOW)
        if win.any():
            pos, d = pos[win], d[win]
            miss_run = 0
        else:
            miss_run += 1
            if miss_run < LABEL_RESET_AFTER:
                continue
            miss_run = 0  # re-anchor on the nearest route arc
        k = np.lexsort((pos, d))[0]
        ptr = int(pos[k])
        true_arc[i], true_d[i] = int(route[ptr]), d[k]
    return true_arc, true_d


def build_trip_table(trip: Trip, with_labels: bool = True) -> tuple[pd.DataFrame, dict]:
    """with_labels=False builds candidates + features only (no route needed): the inference path."""
    P = trip.track[["x", "y"]].to_numpy()
    t = trip.track["t"].to_numpy()
    N = trip.nodes[["x", "y"]].to_numpy()
    arcs = trip.arcs.to_numpy().astype(int)
    n, n_arcs = len(P), len(arcs)

    if with_labels:
        true_arc, true_d = label_points(P, N, arcs, trip.route)
    else:
        true_arc, true_d = np.full(n, -1, dtype=int), np.full(n, np.nan)

    # ---- candidate generation -------------------------------------------------
    pts, aidx = _densify(N, arcs, DENSIFY_STEP_M)
    lists = cKDTree(pts).query_ball_point(P, CAND_RADIUS_M + DENSIFY_STEP_M / 2)
    cnt = np.fromiter((len(l) for l in lists), dtype=int, count=n)
    pt_flat = np.repeat(np.arange(n), cnt)
    arc_flat = aidx[np.concatenate([np.asarray(l, dtype=int) for l in lists])] if cnt.sum() else np.array([], int)
    key = np.unique(pt_flat.astype(np.int64) * n_arcs + arc_flat)
    pt, arc = key // n_arcs, key % n_arcs
    A, B = N[arcs[arc, 0]], N[arcs[arc, 1]]
    d, tf = _seg_dist(P[pt], A, B)
    keep = d <= CAND_RADIUS_M
    pt, arc, d, tf, A, B = pt[keep], arc[keep], d[keep], tf[keep], A[keep], B[keep]
    df = pd.DataFrame({"pt": pt, "cand_arc": arc, "dist": d, "t_frac": tf})
    df = df.sort_values(["pt", "dist"], kind="stable")
    df["dist_rank"] = df.groupby("pt").cumcount()
    df = df[df.dist_rank < MAX_CANDS].reset_index(drop=True)
    pt, arc = df["pt"].to_numpy(), df["cand_arc"].to_numpy()
    A, B = N[arcs[arc, 0]], N[arcs[arc, 1]]
    arc_vec = B - A
    arc_len = np.hypot(*arc_vec.T)
    arc_dir = arc_vec / np.maximum(arc_len, 1e-9)[:, None]

    df["trip_id"] = trip.trip_id
    df["from_node"], df["to_node"] = arcs[arc, 0], arcs[arc, 1]
    df["n_cands"] = df.groupby("pt")["pt"].transform("size")
    df["dist_minus_min"] = df["dist"] - df.groupby("pt")["dist"].transform("min")
    df["arc_len"] = arc_len
    df["along_m"] = df["t_frac"] * arc_len

    # ---- motion features (per point) -----------------------------------------
    idx = np.arange(n)

    def clamp(o):
        return np.clip(idx + o, 0, n - 1)

    def heading(w):
        v = P[clamp(w)] - P[clamp(-w)]
        return v, np.hypot(*v.T)

    v1, m1 = heading(1)
    v3, m3 = heading(3)
    dt_prev = np.diff(t, prepend=t[0])
    dt_prev[0] = 1.0
    dt_next = np.diff(t, append=t[-1])
    dt_next[-1] = 1.0
    dt_prev, dt_next = np.maximum(dt_prev, 1e-3), np.maximum(dt_next, 1e-3)
    step = np.hypot(*(P[clamp(1)] - P[clamp(-1)]).T)
    span = np.maximum(t[clamp(1)] - t[clamp(-1)], 1e-3)
    speed = np.clip(step / span, 0, 60.0)

    def cosh(v, m):
        c = (v[pt] * arc_dir).sum(1) / np.maximum(m[pt], 1e-9)
        return np.where(m[pt] < 2.0, 0.0, c)  # stationary / jittering: no heading info

    df["cos_h1"], df["cos_h3"] = cosh(v1, m1), cosh(v3, m3)
    df["speed"] = speed[pt]
    df["log_dt_prev"] = np.log1p(np.minimum(dt_prev, 600.0))[pt]
    df["log_dt_next"] = np.log1p(np.minimum(dt_next, 600.0))[pt]

    # ---- neighbour-context distances to the same candidate arc -----------------
    ctx = []
    for o in CTX_OFFSETS:
        dd, _ = _seg_dist(P[clamp(o)[pt]], A, B)
        name = f"d_{'m' if o < 0 else 'p'}{abs(o)}"
        df[name] = np.minimum(dd, 100.0)
        ctx.append(name)
    df["d_ctx_mean"] = df[ctx].mean(axis=1)
    df["d_ctx_max"] = df[ctx].max(axis=1)

    # ---- topology -----------------------------------------------------------------
    pairs = set(map(tuple, arcs.tolist()))
    df["has_reverse"] = np.fromiter(
        ((b, a) in pairs for a, b in zip(df.from_node, df.to_node)), dtype=float, count=len(df)
    )
    deg = np.bincount(arcs.ravel(), minlength=len(N))
    df["deg_from"], df["deg_to"] = deg[df.from_node.to_numpy()], deg[df.to_node.to_numpy()]
    # nearest candidate per point (rank 0) -> connectivity of every candidate to the neighbours' best
    best = df[df.dist_rank == 0].set_index("pt")
    bf = np.full(n, -1)
    bt = np.full(n, -1)
    ba = np.full(n, -1)
    bf[best.index], bt[best.index], ba[best.index] = best.from_node, best.to_node, best.cand_arc

    def conn(off, fwd):
        j = np.clip(pt + off, 0, n - 1)
        valid = (ba[j] >= 0) & (j != pt)
        same = arc == ba[j]
        ok = (df.from_node.to_numpy() == bt[j]) if fwd else (df.to_node.to_numpy() == bf[j])
        rev = (df.from_node.to_numpy() == bt[j]) & (df.to_node.to_numpy() == bf[j])
        return (valid & (same | ok)).astype(float), (valid & rev).astype(float)

    df["conn_prev"], df["rev_prev"] = conn(-1, True)
    df["conn_next"], df["rev_next"] = conn(1, False)

    # ---- label -----------------------------------------------------------------------
    df["true_arc"] = true_arc[pt]
    df["pt_has_label"] = df["true_arc"] >= 0
    df["label"] = (df["cand_arc"] == df["true_arc"]).astype(int)
    df["pt_true_dist"] = true_d[pt]

    labelled = true_arc >= 0
    in_set = np.zeros(n, bool)
    in_set[pt[df["label"].to_numpy() == 1]] = True
    info = dict(
        trip_id=trip.trip_id,
        partition=trip.partition,
        n_points=n,
        n_labelled=int(labelled.sum()),
        label_frac=float(labelled.mean()),
        n_no_cand=int(n - df["pt"].nunique()),
        candidate_miss_rate=float(1 - in_set[labelled].mean()) if labelled.any() else np.nan,
        mean_cands=float(len(df) / n),
        max_speed=float(np.clip(np.hypot(*np.diff(P, axis=0).T) / np.maximum(np.diff(t), 1e-3), 0, None).max()),
        max_dt=float(np.diff(t).max()),
    )
    return df[
        ["trip_id", "pt", "cand_arc", "from_node", "to_node", "true_arc", "pt_has_label", "label"] + FEATURES
    ], info
