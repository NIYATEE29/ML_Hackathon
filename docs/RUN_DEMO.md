# Run the full demo (API + UI)

Two terminals. Needs `data/raw/student_test/` (trip folders) and `models/rf.joblib`
(created by `python -m src.models.run_baselines`).

**Terminal 1 - backend (Laptop 1)**
```bash
# optional: PATHSNAP_MODEL=<name in models/> (default rf), PATHSNAP_TRIPS_DIR=<folder with trip folders>
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Check: open http://localhost:8000/health -> `{"status":"ok","model":"rf","n_trips":10}`.
Interactive docs: http://localhost:8000/docs. Every request is logged to `logs/api_requests.log`.

**Terminal 2 - UI (Laptop 2, or the same machine)**
```bash
cd ui
python -m http.server 8080
```
Open http://localhost:8080/?api=http://<laptop-1-ip>:8000 (use `localhost` on one machine),
pick a trip, press **Run**.

**Switching the model later** (once the best one is chosen): restart the backend with
`PATHSNAP_MODEL=<name>`. Nothing else changes. Note the HMM decoder parameters are read from
`models/hmm_params_<name>.json`; re-run `python -m src.hmm.run_hmm_eval` for the new model first.

Contract: `api/CONTRACT.md`. The API computes all metrics server-side; the reference route is
never sent to the UI (tests/test_api.py has a leak test).

`api/legacy_main.py`, `src/hmm/legacy_viterbi.py` and `src/pipeline.py` are the earlier MLP-only
backend; they are kept working but are not used by the demo.
