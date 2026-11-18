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
