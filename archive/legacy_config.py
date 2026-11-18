"""Settings used only by the archived pipelines (kept as they were, pointed at archive/artifacts)."""
from pathlib import Path

from src.config import RAW_DATA_DIR, ROOT, SEED  # noqa: F401  (re-exported for the archived code)

PROJECT_ROOT = ROOT
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODELS_DIR = RESULTS_DIR = PROCESSED_DIR = PROCESSED_DATA_DIR = ARTIFACTS_DIR
FIGURES_DIR = LOGS_DIR = TABLES_DIR = MAPS_DIR = ARTIFACTS_DIR

TRAIN_DIR = RAW_DATA_DIR / "train"
STUDENT_TEST_DIR = RAW_DATA_DIR / "student_test"

N_FOLDS = 5
CANDIDATE_SEARCH_RADIUS_M = 40.0
CANDIDATE_TOP_K = 10
ARC_DENSIFY_STEP_M = 10.0
EARTH_RADIUS_M = 6371000.0

FEATURE_COLUMNS = [
    "dist_to_arc", "proj_ratio", "heading_diff", "speed", "time_delta",
    "arc_length", "dist_rank", "num_candidates", "spatial_likelihood",
]

# Dataset notes found during early exploration (the EDA notebook recomputes these from the data).
PROBLEM_TRIPS = {
    "offset_track_route": ["00000054", "00000033", "00000002"],
    "large_time_gaps": ["00000090", "00000078", "00000069", "00000021"],
    "speed_teleports": ["00000079", "00000013", "00000033", "00000043"],
    "route_discontinuities": [
        "00000002", "00000019", "00000033", "00000041", "00000042",
        "00000069", "00000075", "00000080", "00000086", "00000089", "00000076",
    ],
}
