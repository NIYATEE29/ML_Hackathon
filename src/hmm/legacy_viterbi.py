import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

class PathSnapHMMDecoder:
    """
    Hidden Markov Model (HMM) + Viterbi route reconstruction engine.
    Snaps noisy observation sequences to continuous graph paths:
      - Hidden states: Candidate road arcs at each time step.
      - Emission: Calibrated classifier probability + spatial distance likelihood.
      - Transition: Network shortest-path distance vs. GPS travel distance.
      - Robustness: Epsilon floor for graph discontinuities + large time gap handling.
    """
    def __init__(
        self,
        nodes: np.ndarray,
        arcs: np.ndarray,
        sigma_z: float = 12.0,  # GPS noise standard deviation in meters
        beta: float = 8.0,       # Transition distance penalty scale in meters
        eps_floor: float = 1e-6 # Probability floor for disconnected components
    ):
        self.nodes = nodes
        self.arcs = arcs
        self.sigma_z = sigma_z
        self.beta = beta
        self.eps_floor = eps_floor
        self.log_eps_floor = float(np.log(eps_floor))
        self._source_cache: Dict[int, Dict[int, float]] = {}

        # Build directed NetworkX graph for topology & shortest path lookups
        self.G = nx.DiGraph()
        for idx, (u, v) in enumerate(arcs):
            u_pt = nodes[u]
            v_pt = nodes[v]
            # Euclidean distance in degrees scaled roughly to meters (or local metric)
            length = np.sqrt(np.sum((v_pt - u_pt) ** 2)) * 111000.0
            self.G.add_edge(u, v, arc_id=idx, weight=length)

    def compute_network_distance(self, from_arc: int, to_arc: int) -> float:
        """
        Computes the shortest path distance in the road network
        from the end of from_arc to the start of to_arc (with neighborhood caching).
        """
        if from_arc == to_arc:
            return 0.0

        u_end = int(self.arcs[from_arc, 1])
        v_start = int(self.arcs[to_arc, 0])

        if u_end == v_start:
            return 0.0  # Directly connected consecutive arcs

        if u_end not in self._source_cache:
            try:
                self._source_cache[u_end] = nx.single_source_dijkstra_path_length(
                    self.G, u_end, cutoff=200.0, weight="weight"
                )
            except Exception:
                self._source_cache[u_end] = {}

        return self._source_cache[u_end].get(v_start, 500.0)

    def decode(
        self,
        candidate_lists: List[List[int]],
        candidate_probs: List[List[float]],
        candidate_dists: List[List[float]],
        gps_xy: np.ndarray,
        time_deltas: np.ndarray
    ) -> List[int]:
        """
        Viterbi dynamic programming recurrence to find the globally optimal sequence of arcs.
        """
        T = len(candidate_lists)
        if T == 0:
            return []

        # Forward DP tables
        V = []
        backpointers = []

        # 1. Initialization (t = 0)
        c0 = candidate_lists[0]
        p0 = candidate_probs[0]
        d0 = candidate_dists[0]
        v0 = np.zeros(len(c0))

        for i in range(len(c0)):
            # Log emission = ln(P_classifier) - dist^2 / (2 * sigma^2)
            prob_clf = max(1e-5, p0[i])
            spatial_term = -(d0[i] ** 2) / (2.0 * (self.sigma_z ** 2))
            v0[i] = np.log(prob_clf) + spatial_term

        V.append(v0)
        backpointers.append(np.zeros(len(c0), dtype=int))

        # 2. Viterbi Forward Recurrence (t = 1 to T-1)
        for t in range(1, T):
            prev_cands = candidate_lists[t - 1]
            curr_cands = candidate_lists[t]
            curr_probs = candidate_probs[t]
            curr_dists = candidate_dists[t]

            # Euclidean GPS displacement
            gps_step_dist = float(np.linalg.norm(gps_xy[t] - gps_xy[t - 1]))
            dt = time_deltas[t]

            vt = np.full(len(curr_cands), -np.inf)
            bp_t = np.zeros(len(curr_cands), dtype=int)

            if dt > 30.0:
                # Reset transition penalty after large time gaps (O(1) transition)
                best_prev_idx = int(np.argmax(V[t - 1]))
                best_prev_val = V[t - 1][best_prev_idx]
                for j, c_j in enumerate(curr_cands):
                    prob_clf = max(1e-5, curr_probs[j])
                    spatial_term = -(curr_dists[j] ** 2) / (2.0 * (self.sigma_z ** 2))
                    vt[j] = best_prev_val + np.log(prob_clf) + spatial_term
                    bp_t[j] = best_prev_idx
            else:
                for j, c_j in enumerate(curr_cands):
                    prob_clf = max(1e-5, curr_probs[j])
                    spatial_term = -(curr_dists[j] ** 2) / (2.0 * (self.sigma_z ** 2))
                    log_emission = np.log(prob_clf) + spatial_term

                    best_val = -np.inf
                    best_prev_idx = 0

                    for i, c_i in enumerate(prev_cands):
                        net_dist = self.compute_network_distance(c_i, c_j)
                        diff = abs(net_dist - gps_step_dist)
                        # Exponential transition likelihood: ln(max(eps, exp(-diff/beta)))
                        log_trans = max(self.log_eps_floor, -diff / self.beta)

                        score = V[t - 1][i] + log_trans
                        if score > best_val:
                            best_val = score
                            best_prev_idx = i

                    vt[j] = best_val + log_emission
                    bp_t[j] = best_prev_idx

            V.append(vt)
            backpointers.append(bp_t)

        # 3. Backtracking for Global Optimal Sequence
        opt_seq_indices = [int(np.argmax(V[-1]))]
        for t in range(T - 1, 0, -1):
            prev_idx = backpointers[t][opt_seq_indices[-1]]
            opt_seq_indices.append(prev_idx)

        opt_seq_indices.reverse()
        best_route = [candidate_lists[t][opt_seq_indices[t]] for t in range(T)]

        return best_route
