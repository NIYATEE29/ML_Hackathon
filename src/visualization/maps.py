"""Folium map of one trip: GPS, road network, reference route, predicted route.

Every layer comes from our real implementation: the predicted route is produced by the same
`PathSnap` service the API uses (features -> classifier -> HMM/Viterbi). The reference route is read
only here, for visualisation, from the development partitions (train / student_test).

Layers (toggle in the layer control, top right):
  GPS points          blue dots + thin line of the observed track
  Road network        grey, only arcs near the track (a whole network can have tens of thousands of arcs)
  Reference route     green, thick   (ground truth, where a .route file exists)
  Predicted (HMM)     red            (classifier + HMM/Viterbi decoding)
  Classifier only     orange dashed  (highest-probability candidate per point, no HMM)
  HMM changed choice  purple dots    (points where the HMM overrode the classifier's top candidate)

Usage:
  python -m src.visualization.maps --partition student_test --trip 00000052
  python -m src.visualization.maps --partition student_test --all
"""

import argparse
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from api.service import PathSnap, arcs_to_paths
from src.config import RAW_DIR, RESULTS_DIR
from src.data.loader import load_trip

COLORS = dict(gps="#1f6fd1", net="#9a9a9a", ref="#1baf7a", pred="#e0302c", clf="#eda100", changed="#7a3fd0")
MAX_GPS_MARKERS = 1500  # draw every n-th point as a marker above this; the track line is always full


def _network_near(trip, pad_deg=0.004):
    """Network arcs with an endpoint inside the (padded) bounding box of the track."""
    lo_lon, hi_lon = trip.track.lon.min() - pad_deg, trip.track.lon.max() + pad_deg
    lo_lat, hi_lat = trip.track.lat.min() - pad_deg, trip.track.lat.max() + pad_deg
    n = trip.nodes
    inside = ((n.lon >= lo_lon) & (n.lon <= hi_lon) & (n.lat >= lo_lat) & (n.lat <= hi_lat)).to_numpy()
    a = trip.arcs.to_numpy().astype(int)
    keep = inside[a[:, 0]] | inside[a[:, 1]]
    lon, lat = n.lon.to_numpy(), n.lat.to_numpy()
    return [[[lat[f], lon[f]], [lat[t], lon[t]]] for f, t in a[keep]], int(keep.sum()), len(a)


def _legend(rows):
    items = "".join(
        f'<div><span style="display:inline-block;width:22px;height:0;border-top:4px {s} {c};'
        f'vertical-align:middle;margin-right:6px"></span>{t}</div>'
        for c, s, t in rows
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:8px 12px;'
        'border:1px solid #bbb;border-radius:6px;font:12px/1.5 sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.25)">'
        f"<b>Legend</b>{items}</div>"
    )
    return folium.Element(html)


def build_map(partition: str, trip_id: str, model: str = "xgb_tuned", service: PathSnap | None = None) -> folium.Map:
    raw_partition_dir = RAW_DIR / partition
    svc = service or PathSnap(raw_partition_dir, model)
    res = svc.predict(trip_id)
    trip = load_trip(partition, trip_id, raw_dir=RAW_DIR, with_route=True)  # route only for display

    gps = np.asarray(res["gps"])
    m = folium.Map(location=gps.mean(axis=0).tolist(), zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri)",
        show=False,
    ).add_to(m)

    net_lines, n_near, n_all = _network_near(trip)
    g = folium.FeatureGroup(name=f"Road network ({n_near:,} of {n_all:,} arcs near the track)", show=True)
    folium.PolyLine(net_lines, color=COLORS["net"], weight=1.5, opacity=0.7).add_to(g)
    g.add_to(m)

    rows = [(COLORS["net"], "solid", "Road network (near track)")]
    if len(trip.route):
        ref_paths = arcs_to_paths(trip.route.tolist(), trip.arcs.to_numpy().astype(int), trip.nodes)
        g = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        for p in ref_paths:
            folium.PolyLine(p, color=COLORS["ref"], weight=8, opacity=0.55).add_to(g)
        g.add_to(m)
        rows.append((COLORS["ref"], "solid", "Reference route"))

    g = folium.FeatureGroup(name="Classifier-only route", show=False)
    for p in res["classifier_route"]["paths"]:
        folium.PolyLine(p, color=COLORS["clf"], weight=4, opacity=0.9, dash_array="8 6").add_to(g)
    g.add_to(m)
    rows.append((COLORS["clf"], "dashed", "Classifier only"))

    g = folium.FeatureGroup(name=f"Predicted route ({res['model']} + HMM)", show=True)
    for p in res["predicted_route"]["paths"]:
        folium.PolyLine(p, color=COLORS["pred"], weight=4, opacity=0.9).add_to(g)
    g.add_to(m)
    rows.append((COLORS["pred"], "solid", f"Predicted: {res['model']} + HMM"))

    g = folium.FeatureGroup(name="GPS observations", show=True)
    folium.PolyLine(gps.tolist(), color=COLORS["gps"], weight=1, opacity=0.5).add_to(g)
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    for i in range(0, len(gps), step):
        folium.CircleMarker(
            gps[i].tolist(),
            radius=2,
            color=COLORS["gps"],
            fill=True,
            fill_opacity=0.9,
            weight=0,
            tooltip=f"GPS point {i}",
        ).add_to(g)
    folium.CircleMarker(
        gps[0].tolist(),
        radius=8,
        color="#ffffff",
        weight=2,
        fill=True,
        fill_color="#1a7f37",
        fill_opacity=1,
        tooltip="start",
    ).add_to(g)
    folium.CircleMarker(
        gps[-1].tolist(),
        radius=8,
        color="#ffffff",
        weight=2,
        fill=True,
        fill_color="#111111",
        fill_opacity=1,
        tooltip="end",
    ).add_to(g)
    g.add_to(m)
    rows.append((COLORS["gps"], "dotted", "GPS observations"))

    cands, dec = res["_cands"], res["_decoded"]  # where did the HMM overrule the classifier?
    clf = cands.loc[cands.groupby("pt")["p"].idxmax()].set_index("pt").cand_arc
    hmm = pd.Series(dec.cand_arc.to_numpy(), index=dec.pt.to_numpy()) if "pt" in dec else None
    if hmm is not None:
        both = pd.concat([clf.rename("clf"), hmm.rename("hmm")], axis=1).dropna()
        changed = both.index[both.clf != both.hmm].to_numpy()
        g = folium.FeatureGroup(name=f"Points where HMM changed the classifier's choice ({len(changed)})", show=False)
        for i in changed[:3000]:
            folium.CircleMarker(
                gps[int(i)].tolist(),
                radius=3,
                color=COLORS["changed"],
                fill=True,
                fill_opacity=0.9,
                weight=0,
                tooltip=f"point {int(i)}: classifier arc {int(both.clf[i])} -> HMM arc {int(both.hmm[i])}",
            ).add_to(g)
        g.add_to(m)
        rows.append((COLORS["changed"], "dotted", "HMM changed classifier choice"))

    m.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()], padding=(40, 40))
    folium.LayerControl(collapsed=True).add_to(m)
    m.get_root().html.add_child(_legend(rows))
    m.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:#fff;padding:4px 10px;border:1px solid #bbb;'
            f'border-radius:6px;font:13px sans-serif">PathSnap trip {trip_id} ({partition}), {len(gps):,} GPS points</div>'
        )
    )
    return m


def save_map(
    partition: str,
    trip_id: str,
    model: str = "xgb_tuned",
    out_dir: Path = RESULTS_DIR / "maps",
    service: PathSnap | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{partition}_{trip_id}_prediction.html"
    build_map(partition, trip_id, model, service).save(str(path))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trip")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="xgb_tuned")
    a = ap.parse_args()
    svc = PathSnap(RAW_DIR / a.partition, a.model)
    for t in svc.trip_ids() if a.all else [a.trip]:
        print(save_map(a.partition, t, a.model, service=svc))
