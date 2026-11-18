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

from src.config import PROJECTED_CRS, RAW_DIR

_to_m = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame   # lon, lat, t, x, y
    nodes: pd.DataFrame   # lon, lat, x, y
    arcs: pd.DataFrame    # from_node, to_node
    route: np.ndarray     # arc ids


def _read(path: Path, names, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def _add_xy(df: pd.DataFrame) -> pd.DataFrame:
    x, y = _to_m.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    return df


def list_trips(partition: str, raw_dir: Path = RAW_DIR) -> list[str]:
    return sorted(p.name for p in (raw_dir / partition).iterdir() if p.is_dir())


def load_trip(partition: str, trip_id: str, raw_dir: Path = RAW_DIR) -> Trip:
    d = raw_dir / partition / trip_id
    track = _add_xy(_read(d / f"{trip_id}.track", ["lon", "lat", "t"]))
    nodes = _add_xy(_read(d / f"{trip_id}.nodes", ["lon", "lat"]))
    arcs = _read(d / f"{trip_id}.arcs", ["from_node", "to_node"], dtype=int)
    route = _read(d / f"{trip_id}.route", ["arc_id"], dtype=int)["arc_id"].to_numpy()
    return Trip(trip_id, partition, track, nodes, arcs, route)
