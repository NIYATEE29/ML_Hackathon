# PathSnap API contract (v1)

The UI talks to the backend ONLY through these three endpoints. Real example responses (generated
by the actual pipeline) are in `ui/mock/`. Build and test the UI against those files first.

Base URL: `http://<laptop-1-ip>:8000` (CORS is open, so a page served from Laptop 2 can call it).
All bodies are JSON. Coordinates are always `[lat, lon]` (Leaflet order).

## GET /health
```json
{"status": "ok", "model": "rf", "n_trips": 10}
```
Use it for a "connected / not connected" indicator. Example: `ui/mock/health.json`.

## GET /trips
The trips the evaluator can choose from (the 10 reserved trips on the day).
```json
{"trips": [{"trip_id": "00000000", "n_points": 2503}, ...]}
```
Example: `ui/mock/trips.json`. `trip_id` is an 8-digit string. Keep it a string (leading zeros).

## POST /predict
Request: `{"trip_id": "00000052"}`

Runs the whole pipeline (features, classifier, HMM/Viterbi) and returns:
```json
{
  "trip_id": "00000052",
  "model": "rf",
  "decoder": "hmm_viterbi",
  "n_points": 262,
  "gps": [[lat, lon], ...],                       // every raw GPS point, in time order (length n_points)
  "predicted_route": {                            // FINAL route (classifier + HMM): draw this one prominently
    "arcs": [571, 569, ...],                      // road-segment ids, consecutive duplicates removed
    "paths": [[[lat, lon], [lat, lon], ...], ...] // polylines; a NEW path starts where the route is disconnected
  },
  "classifier_route": { "arcs": [...], "paths": [...] },   // classifier-only route (optional comparison layer)
  "metrics": {
    "final":           {"top1_accuracy": 1.0,    "macro_f1": 1.0,    "route_continuity": 0.9655, "route_f1": 0.9326, "n_scored_points": 262, "candidate_miss_rate": 0.0},
    "classifier_only": {"top1_accuracy": 0.9695, "macro_f1": 0.9821, "route_continuity": 0.954,  "route_f1": 0.8715, "n_scored_points": 262, "candidate_miss_rate": 0.0}
  },
  "timing_ms": {"load": 44, "features_and_classifier": 115, "hmm": 21, "total": 181}
}
```
Examples: `ui/mock/predict_00000052.json` (small), `ui/mock/predict_00000065.json` (larger).

Field notes
- `paths` is a list of polylines. Pass it straight to `L.polyline(paths)` (Leaflet draws multi-polylines).
  Do NOT join the paths into one line: the gaps are real breaks in the predicted route.
- `metrics` values are fractions in 0..1 (show as percentages if you like). `final` is what the
  evaluator cares about. Show `classifier_only` next to it so the HMM improvement is visible.
- `metrics` can be `null` if the server has no reference route for that trip. The UI must handle it
  (show "n/a") and must not crash.
- `timing_ms` is server-side processing time. The UI may also show its own round-trip time.

Errors
- Unknown trip: HTTP 404, body `{"detail": "unknown trip_id '99999999'"}` (`ui/mock/error_404.json`).
- Server unreachable or other failure: show a clear message on screen (not just the console).

## Rules for the UI
- The UI must NEVER receive or display the reference ("ground truth") route. The API does not send it.
  Do not look for it, request it, or add a field for it.
- The UI must not compute accuracy itself. It displays the numbers in `metrics`.
- The evaluator selects one of the trips, the UI sends `trip_id`, the API runs the real pipeline.
  Nothing is cached or hardcoded in the UI for specific trips (mock files are for development only).
- Make the API address configurable (a text box or a `?api=` URL parameter, default `http://localhost:8000`)
  because the API runs on a different laptop on the day.

## Running the backend (Laptop 1)
```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Environment: `PATHSNAP_TRIPS_DIR` (folder holding the trip folders, default `data/raw/student_test`),
`PATHSNAP_MODEL` (default `rf`). Every request is printed to the console and written to
`logs/api_requests.log`, which is what we show the evaluator.
