import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive.legacy_config import STUDENT_TEST_DIR, TRAIN_DIR, MODELS_DIR
from src.data.loader import list_trips
from archive.legacy_loader import load_trip_data
from archive.abhijith_mlp.pipeline import PathSnapPipeline

app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Distributed FastAPI backend (Laptop 1) for real-time candidate scoring, MLP inference, and HMM Viterbi route reconstruction.",
    version="1.0.0"
)

# Enable CORS for cross-laptop communication (Laptop 1 backend <-> Laptop 2 UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy-loaded or initialized on startup)
pipeline: Optional[PathSnapPipeline] = None

def get_pipeline() -> PathSnapPipeline:
    global pipeline
    if pipeline is None:
        model_path = MODELS_DIR / "mlp_model.pkl"
        pipeline = PathSnapPipeline(model_path=model_path)
    return pipeline

class TripPredictionRequest(BaseModel):
    trip_id: str
    split: str = "student_test"
    use_cached_features: bool = True

class TripPredictionResponse(BaseModel):
    trip_id: str
    split: str
    num_points: int
    num_candidates: int
    predicted_route: List[int]
    classifier_predicted_route: List[int]
    classifier_continuity: float
    hmm_continuity: float
    timings: Dict[str, float]

class TripEvaluationResponse(BaseModel):
    trip_id: str
    split: str
    num_points: int
    classifier_top1_accuracy: Optional[float] = None
    hmm_top1_accuracy: Optional[float] = None
    classifier_continuity: float
    hmm_continuity: float
    route_jaccard_similarity: Optional[float] = None
    timings: Dict[str, float]

@app.on_event("startup")
def startup_event():
    try:
        get_pipeline()
    except Exception as e:
        print(f"Warning: Pipeline initialization deferred: {e}")

@app.get("/")
def root():
    pipe = get_pipeline()
    return {
        "status": "online",
        "service": "PathSnap ML Reconstruction Backend",
        "model_loaded": pipe.model is not None,
        "supported_models": ["MLP (Multi-Layer Perceptron)"],
        "endpoints": [
            "GET /trips",
            "POST /predict",
            "POST /evaluate"
        ]
    }

@app.get("/trips")
def get_available_trips(split: str = "student_test"):
    """
    Returns list of available evaluation trips without leaking ground truth route labels.
    """
    trips = list_trips(split)
    trip_metadata = []
    for tid in trips:
        try:
            data = load_trip_data(tid, split)
            track = data["track"]
            nodes = data["nodes"]
            arcs = data["arcs"]
            trip_metadata.append({
                "trip_id": tid,
                "split": split,
                "num_gps_points": len(track),
                "num_network_arcs": len(arcs),
                "num_network_nodes": len(nodes),
                "start_time_s": float(track[0, 2]) if len(track) > 0 else 0.0,
                "duration_s": float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0
            })
        except Exception as e:
            continue
    return {"count": len(trip_metadata), "trips": trip_metadata}

@app.post("/predict", response_model=TripPredictionResponse)
def predict_route(req: TripPredictionRequest):
    """
    Executes the full pipeline:
      1. Dynamic candidate generation
      2. MLP classification
      3. HMM Viterbi decoding
    Returns the predicted route and timing metrics.
    """
    pipe = get_pipeline()
    try:
        res = pipe.run_trip(
            trip_id=req.trip_id,
            split=req.split,
            use_cached_features=req.use_cached_features
        )
        return TripPredictionResponse(
            trip_id=res["trip_id"],
            split=res["split"],
            num_points=res["num_points"],
            num_candidates=res["num_candidates"],
            predicted_route=res["hmm_predicted_route"],
            classifier_predicted_route=res["classifier_predicted_route"],
            classifier_continuity=res["classifier_continuity"],
            hmm_continuity=res["hmm_continuity"],
            timings=res["timings"]
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/evaluate", response_model=TripEvaluationResponse)
def evaluate_trip(req: TripPredictionRequest):
    """
    Strictly Server-Side Evaluation:
    Evaluates predictions against ground truth without exposing ground truth labels to the UI.
    """
    pipe = get_pipeline()
    try:
        res = pipe.run_trip(
            trip_id=req.trip_id,
            split=req.split,
            use_cached_features=req.use_cached_features
        )
        return TripEvaluationResponse(
            trip_id=res["trip_id"],
            split=res["split"],
            num_points=res["num_points"],
            classifier_top1_accuracy=res.get("classifier_top1_accuracy"),
            hmm_top1_accuracy=res.get("hmm_top1_accuracy"),
            classifier_continuity=res["classifier_continuity"],
            hmm_continuity=res["hmm_continuity"],
            route_jaccard_similarity=res.get("route_jaccard_similarity"),
            timings=res["timings"]
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
