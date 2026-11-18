"""Load PathSnap trips from the raw files.

Each trip directory holds four whitespace-separated, header-less files:
  .track  : lon, lat, timestamp(s)
  .nodes  : lon, lat            (row index = node_id)
  .arcs   : from_node, to_node  (row index = arc_id)
  .route  : arc_id              (reference route, chronological)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from src.config import RAW_DATA_DIR, RAW_DIR


def _local_transformer(lon0: float, lat0: float) -> Transformer:
    """Per-trip azimuthal-equidistant projection (metres) centred on the trip.

    Trips are spread across the whole world (lon -100..137, lat -35..62), so no single
    fixed CRS is valid. Distances are near-exact close to the trip centre.
    """
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame  # lon, lat, t, x, y
    nodes: pd.DataFrame  # lon, lat, x, y
    arcs: pd.DataFrame  # from_node, to_node
    route: np.ndarray  # arc ids


def _read(path: Path, names, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def _add_xy(df: pd.DataFrame, tf: Transformer) -> pd.DataFrame:
    x, y = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    return df


def _resolve_raw_dir(raw_dir: Path, partition: str) -> Path:
    if (raw_dir / partition).exists():
        return raw_dir
    if (RAW_DATA_DIR / partition).exists():
        return RAW_DATA_DIR
    return raw_dir


def list_trips(partition: str, raw_dir: Path = RAW_DIR) -> list[str]:
    base = _resolve_raw_dir(raw_dir, partition)
    return sorted(p.name for p in (base / partition).iterdir() if p.is_dir())


def load_trip(partition: str, trip_id: str, raw_dir: Path = RAW_DIR, with_route: bool = True) -> Trip:
    """with_route=False never opens the .route file (inference must not see ground truth)."""
    base = _resolve_raw_dir(raw_dir, partition)
    d = base / partition / trip_id
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"])
    tf = _local_transformer(track["lon"].mean(), track["lat"].mean())
    track = _add_xy(track, tf)
    nodes = _add_xy(_read(d / f"{trip_id}.nodes", ["lon", "lat"]), tf)
    arcs = _read(d / f"{trip_id}.arcs", ["from_node", "to_node"], dtype=int)
    route_file = d / f"{trip_id}.route"
    if with_route and route_file.exists():
        route = _read(route_file, ["arc_id"], dtype=int)["arc_id"].to_numpy()
    else:
        route = np.empty(0, dtype=int)
    return Trip(trip_id, partition, track, nodes, arcs, route)
