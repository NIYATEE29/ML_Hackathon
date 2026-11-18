"""Central configuration: paths, seeds and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42
# NOTE: trips are located all over the world, so there is no single projected CRS.
# src/data/loader.py projects each trip with its own local azimuthal-equidistant CRS.

# Backwards-compatible aliases and derived paths
PROJECT_ROOT = ROOT
if (ROOT / "Student_Folder").exists():
    RAW_DATA_DIR = ROOT / "Student_Folder"
else:
    RAW_DATA_DIR = RAW_DIR

TRAIN_DIR = RAW_DATA_DIR / "train"
STUDENT_TEST_DIR = RAW_DATA_DIR / "student_test"
PROCESSED_DATA_DIR = PROCESSED_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
TABLES_DIR = RESULTS_DIR / "tables"
MAPS_DIR = RESULTS_DIR / "maps"

for _d in [FIGURES_DIR, LOGS_DIR, TABLES_DIR, MAPS_DIR, MODELS_DIR, PROCESSED_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CANDIDATE_SEARCH_RADIUS_M = 40.0
CANDIDATE_TOP_K = 10
ARC_DENSIFY_STEP_M = 10.0
EARTH_RADIUS_M = 6371000.0

FEATURE_COLUMNS = [
    "dist_to_arc",
    "proj_ratio",
    "heading_diff",
    "speed",
    "time_delta",
    "arc_length",
    "dist_rank",
    "num_candidates",
    "spatial_likelihood"
]

PROBLEM_TRIPS = {
    "offset_track_route": ["00000054", "00000033", "00000002"],
    "large_time_gaps": ["00000090", "00000078", "00000069", "00000021"],
    "speed_teleports": ["00000079", "00000013", "00000033", "00000043"],
    "route_discontinuities": [
        "00000002", "00000019", "00000033", "00000041", "00000042",
        "00000069", "00000075", "00000080", "00000086", "00000089", "00000076"
    ]
}

EVALUATOR_TRIP_IDS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]
