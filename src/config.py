"""Central configuration: paths, seeds and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42
PROJECTED_CRS = "EPSG:32630"  # UTM zone 30N (study area is ~ lon -1.8, lat 53.0)
