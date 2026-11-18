"""Paths and constants shared by the whole project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Some teammates keep the unzipped data in a "Student_Folder" next to the code instead of data/raw.
# The loader falls back to RAW_DATA_DIR when a partition is not found under data/raw.
RAW_DATA_DIR = ROOT / "Student_Folder" if (ROOT / "Student_Folder").exists() else RAW_DIR

# Trips are located all over the world, so there is no single projected CRS.
# src/data/loader.py projects each trip with its own local azimuthal-equidistant CRS.
