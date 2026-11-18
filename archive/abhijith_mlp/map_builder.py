import os
from pathlib import Path
from typing import List, Optional
import folium
import numpy as np

from archive.legacy_config import MAPS_DIR
from archive.legacy_loader import load_trip_data

def build_interactive_folium_map(
    trip_id: str,
    split: str = "train",
    predicted_arcs: Optional[List[int]] = None,
    save_html_path: Optional[Path] = None
) -> folium.Map:
    """
    Builds a high-performance multi-layer interactive Folium map visualization:
      - Layer 1: Road Network Arcs (Grey lines within 500m of trajectory)
      - Layer 2: Raw Noisy GPS Trajectory (Blue circles + connecting dashed line)
      - Layer 3: Ground Truth Reference Route (Red solid line)
      - Layer 4: Model Predicted Route (Green solid line, if provided)
    """
    data = load_trip_data(trip_id, split)
    nodes = data["nodes"]
    arcs = data["arcs"]
    track = data["track"]
    route = data["route"]

    center_lat = float(np.median(track[:, 1]))
    center_lon = float(np.median(track[:, 0]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # 1. Road Network Layer (vectorized bounding-box filtering for instant rendering)
    road_layer = folium.FeatureGroup(name="Road Network Graph", show=False)
    min_lon, max_lon = track[:, 0].min(), track[:, 0].max()
    min_lat, max_lat = track[:, 1].min(), track[:, 1].max()
    pad = 0.005  # ~500m

    u_nodes = arcs[:, 0]
    v_nodes = arcs[:, 1]
    u_xy = nodes[u_nodes]
    v_xy = nodes[v_nodes]

    mask = (
        (u_xy[:, 0] >= min_lon - pad) & (u_xy[:, 0] <= max_lon + pad) &
        (u_xy[:, 1] >= min_lat - pad) & (u_xy[:, 1] <= max_lat + pad)
    ) | (
        (v_xy[:, 0] >= min_lon - pad) & (v_xy[:, 0] <= max_lon + pad) &
        (v_xy[:, 1] >= min_lat - pad) & (v_xy[:, 1] <= max_lat + pad)
    )

    filtered_arcs = arcs[mask]
    # Cap to max 2000 arcs to keep browser smooth
    if len(filtered_arcs) > 2000:
        filtered_arcs = filtered_arcs[:2000]

    for u, v in filtered_arcs:
        folium.PolyLine(
            locations=[[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]],
            color="#555555",
            weight=1.5,
            opacity=0.5
        ).add_to(road_layer)
    road_layer.add_to(m)

    # 2. Raw GPS Observation Layer
    gps_layer = folium.FeatureGroup(name="Raw GPS Trajectory (Noisy)", show=True)
    gps_coords = [[lat, lon] for lon, lat, _ in track]
    folium.PolyLine(
        locations=gps_coords,
        color="#0066cc",
        weight=2.5,
        dash_array="6, 6",
        opacity=0.7,
        tooltip="GPS Trajectory"
    ).add_to(gps_layer)

    step = max(1, len(track) // 150)
    for idx in range(0, len(track), step):
        lon, lat, ts = track[idx]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3 if (idx != 0 and idx != len(track) - 1) else 6,
            color="#0044aa" if (idx != 0 and idx != len(track) - 1) else ("#00cc00" if idx == 0 else "#cc0000"),
            fill=True,
            fill_opacity=0.8,
            popup=f"GPS #{idx} | Time: {ts:.1f}s"
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # 3. Reference Route Layer (Ground Truth)
    if route is not None and len(route) > 0:
        gt_layer = folium.FeatureGroup(name="Ground Truth Route (Reference)", show=True)
        for arc_id in route:
            u, v = arcs[arc_id]
            folium.PolyLine(
                locations=[[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]],
                color="#e41a1c",
                weight=4.5,
                opacity=0.85,
                tooltip=f"Ground Truth Arc #{arc_id}"
            ).add_to(gt_layer)
        gt_layer.add_to(m)

    # 4. Model Predicted Route Layer
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        pred_layer = folium.FeatureGroup(name="Predicted Route (MLP)", show=True)
        for arc_id in predicted_arcs:
            u, v = arcs[arc_id]
            folium.PolyLine(
                locations=[[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]],
                color="#2ca02c",
                weight=3,
                opacity=0.9,
                tooltip=f"Predicted Arc #{arc_id}"
            ).add_to(pred_layer)
        pred_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    if save_html_path is None:
        save_html_path = MAPS_DIR / f"trip_{trip_id}_map.html"

    save_html_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(save_html_path))
    print(f"Saved interactive Folium map to {save_html_path}")
    return m

if __name__ == "__main__":
    build_interactive_folium_map("00000001", "train")
