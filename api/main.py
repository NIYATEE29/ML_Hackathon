"""PathSnap FastAPI backend (Laptop 1).

Run:   uvicorn api.main:app --host 0.0.0.0 --port 8000
Config (environment variables):
    PATHSNAP_TRIPS_DIR   folder that directly contains the trip folders
                         (default: data/raw/student_test; on the day: the organisers' reserved trips)
    PATHSNAP_MODEL       saved model name in models/ (default: xgb_tuned)
Contract: api/CONTRACT.md.  Every request is logged to the console and to logs/api_requests.log.
"""

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.scoring import score_against_reference
from api.service import PathSnap
from src.config import ROOT

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "raw" / "student_test"))
MODEL = os.environ.get("PATHSNAP_MODEL", "xgb_tuned")

(ROOT / "logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "api_requests.log")):
        h.setFormatter(fmt)
        log.addHandler(h)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)  # UI runs on another laptop
service = PathSnap(TRIPS_DIR, MODEL)


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info(
        f"{request.client.host if request.client else '-'} | {request.method} {request.url.path} | {response.status_code} | {1000 * (time.perf_counter() - t0):.0f} ms"
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": service.model_name, "n_trips": len(service.trip_ids())}


@app.get("/trips")
def trips():
    return {"trips": service.list_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in service.trip_ids():
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{req.trip_id}'")
    res = service.predict(req.trip_id)  # never sees ground truth
    metrics = score_against_reference(TRIPS_DIR, req.trip_id, res["_cands"], res["_decoded"])  # server side only
    body = {k: v for k, v in res.items() if not k.startswith("_")}
    body["metrics"] = metrics
    log.info(
        f"predict trip={req.trip_id} model={service.model_name} points={body['n_points']} "
        f"top1={metrics['final']['top1_accuracy'] if metrics else 'n/a'} total={body['timing_ms']['total']} ms"
    )
    return body
