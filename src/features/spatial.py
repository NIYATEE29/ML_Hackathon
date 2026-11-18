import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from src.config import EARTH_RADIUS_M, CANDIDATE_SEARCH_RADIUS_M, CANDIDATE_TOP_K, ARC_DENSIFY_STEP_M

def to_local_xy(lons: np.ndarray, lats: np.ndarray, center_lon: float, center_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects (lon, lat) degrees to local Cartesian (x, y) meters using Equirectangular projection
    centered at (center_lon, center_lat).
    Accurate to < 0.01% for trip scales (< 100 km) and works globally across any latitude.
    """
    phi = np.radians(center_lat)
    x = EARTH_RADIUS_M * np.radians(lons - center_lon) * np.cos(phi)
    y = EARTH_RADIUS_M * np.radians(lats - center_lat)
    return x, y

def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Computes orthogonal distance from point p to segment ab, and the normalized projection ratio t in [0, 1].
    p, a, b are 2D vectors [x, y].
    Returns: (distance_meters, projection_ratio)
    """
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq <= 1e-12:
        return float(np.linalg.norm(ap)), 0.0

    t = np.dot(ap, ab) / ab_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    projection = a + t_clamped * ab
    dist = np.linalg.norm(p - projection)
    return float(dist), float(t_clamped)

def angle_diff(theta1: float, theta2: float) -> float:
    """Computes the absolute acute angular difference in [0, pi] radians."""
    diff = np.abs((theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi)
    return float(diff)

class TripSpatialIndex:
    """
    Spatial candidate search engine for a single trip road network.
    Constructs a KD-Tree over densified arc points to find candidate arcs in O(log N) time.
    """
    def __init__(self, nodes: np.ndarray, arcs: np.ndarray, densify_step_m: float = ARC_DENSIFY_STEP_M):
        self.nodes = nodes
        self.arcs = arcs
        self.num_arcs = len(arcs)

        # Center on road network median coordinates
        self.center_lon = float(np.median(nodes[:, 0]))
        self.center_lat = float(np.median(nodes[:, 1]))

        # Project all nodes to local (x, y) in meters
        self.node_x, self.node_y = to_local_xy(nodes[:, 0], nodes[:, 1], self.center_lon, self.center_lat)
        self.node_xy = np.column_stack([self.node_x, self.node_y])

        # Compute arc geometries: endpoints, length, bearing angle
        from_xy = self.node_xy[arcs[:, 0]]
        to_xy = self.node_xy[arcs[:, 1]]
        delta = to_xy - from_xy
        self.arc_lengths = np.sqrt(np.sum(delta ** 2, axis=1))
        self.arc_bearings = np.arctan2(delta[:, 1], delta[:, 0])

        # Densify arcs every ~densify_step_m
        pts_list = []
        arc_ids_list = []
        for arc_id in range(self.num_arcs):
            start = from_xy[arc_id]
            end = to_xy[arc_id]
            length = self.arc_lengths[arc_id]
            n_samples = max(2, int(np.ceil(length / densify_step_m)) + 1)
            t = np.linspace(0.0, 1.0, n_samples)[:, None]
            samples = start + t * (end - start)
            pts_list.append(samples)
            arc_ids_list.append(np.full(n_samples, arc_id, dtype=np.int64))

        all_densified_pts = np.vstack(pts_list)
        self.densified_arc_ids = np.concatenate(arc_ids_list)
        self.tree = cKDTree(all_densified_pts)

    def project_gps(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Projects GPS coordinate arrays to local metric coordinates [x, y]."""
        x, y = to_local_xy(lons, lats, self.center_lon, self.center_lat)
        return np.column_stack([x, y])

    def find_candidates(self, query_pt_xy: np.ndarray, radius_m: float = CANDIDATE_SEARCH_RADIUS_M, top_k: int = CANDIDATE_TOP_K) -> List[int]:
        """
        Finds all candidate road arc IDs within radius_m of query_pt_xy.
        Falls back to nearest arc if search ball is empty.
        """
        indices = self.tree.query_ball_point(query_pt_xy, r=radius_m)
        if len(indices) == 0:
            _, nearest_idx = self.tree.query(query_pt_xy, k=1)
            indices = [int(nearest_idx)]

        # Unique arc IDs preserving spatial proximity
        candidate_arcs = []
        seen = set()
        for idx in indices:
            arc_id = int(self.densified_arc_ids[idx])
            if arc_id not in seen:
                seen.add(arc_id)
                candidate_arcs.append(arc_id)
                if len(candidate_arcs) >= top_k:
                    break

        return candidate_arcs
