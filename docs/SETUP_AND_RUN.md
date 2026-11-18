# PathSnap - setup and run guide (every laptop)

Windows PowerShell commands. Run them from the repo root unless a step says otherwise.
Three situations, pick the one you need:

| I want to... | Do |
|---|---|
| Get the project working on a new laptop | Part 1 |
| Run the demo (API + UI), one laptop or two | Part 2 / Part 3 |
| Retrain / re-run the models and comparison | Part 4 |

---

## Part 1 - One-time setup (each teammate, each laptop)

**1. Needs:** Python 3.10+ and Git.

**2. Get the code**
```powershell
git clone https://github.com/NIYATEE29/ML_Hackathon.git
cd ML_Hackathon
```
Already cloned? `git checkout main` then `git pull`.
If `git pull` says local files "would be overwritten" and they are generated outputs
(`results/figures/cmp_*.png`, `results/tables/model_comparison.*`), run
`git restore results/figures results/tables` and pull again.

**3. Virtual environment + packages**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
Every new terminal needs the `Activate.ps1` line again (you will see `(.venv)` in the prompt).

**4. Put the data in place** (the data is NOT in git - get the `Student_folder.zip` from the team)

Unzip it so the folders look exactly like this:
```
ML_Hackathon\
  data\raw\train\<trip folders like 00000001>\
  data\raw\student_test\<trip folders like 00000009>\
```
Each trip folder contains `<id>.track`, `.nodes`, `.arcs` (and `.route` for train and student_test).
Quick check:
```powershell
dir data\raw\student_test
```
should list 10 trip folders.

**5. Check everything works**
```powershell
python -m pytest tests -q
```
Expect all tests to pass (about 19).

The trained models (`models\*.joblib`) are committed to the repo, so you can run the demo
right now without training anything.

---

## Part 2 - Run the demo on ONE laptop

Open **two** PowerShell windows in the repo root.

**Window 1 - backend**
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Wait for "Application startup complete". Check http://localhost:8000/health in a browser; you should see
`{"status":"ok","model":"xgb_tuned","n_trips":10}`.

**Window 2 - UI**
```powershell
cd ui
python -m http.server 8080
```
Open **http://localhost:8080/?api=http://localhost:8000**, choose a trip, press **Run**.
You should see the GPS points, the predicted route, and the metrics (Top-1, Macro F1, route continuity, route F1).

Stop either window with `Ctrl+C`.

---

## Part 3 - Demo on TWO laptops (the real setup)

- **Laptop 1 = backend** (needs the repo, the venv, `data\raw\student_test`, and `models\`).
- **Laptop 2 = UI** (needs only the `ui\` folder and a browser; it does not need the data or the models).
- Both laptops must be on the **same Wi-Fi / network** (a phone hotspot works well; some college Wi-Fi blocks laptop-to-laptop traffic).

**On Laptop 1**

1. Find its IP address:
   ```powershell
   ipconfig
   ```
   Use the "IPv4 Address" of the Wi-Fi adapter, e.g. `192.168.1.20`.
2. Allow the port through Windows Firewall (once; PowerShell **as Administrator**):
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```
3. Start the backend (same command as Part 2, it must say `--host 0.0.0.0`):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

**On Laptop 2**

1. Check it can reach Laptop 1: open `http://192.168.1.20:8000/health` in a browser (use Laptop 1's real IP).
   It must show `"status":"ok"`. If not, see Troubleshooting.
2. Serve the UI:
   ```powershell
   cd ui
   python -m http.server 8080
   ```
3. Open **http://localhost:8080/?api=http://192.168.1.20:8000** (Laptop 1's IP), pick a trip, press **Run**.
   You can also type the address into the "API address" box on the page and press **Connect**.
   The pill in the header turns green when `/health` works.

**Evaluation day (their trips instead of ours):** on Laptop 1, point the backend at the folder the
evaluators give you. It must be the folder that *directly contains* the trip folders:
```powershell
$env:PATHSNAP_TRIPS_DIR = "C:\path\to\evaluator_trips"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
The trip dropdown on Laptop 2 then lists those trips. Trips without a `.route` file still work
(predictions are shown; metrics show as n/a).

---

## Part 4 - Rebuild features / retrain / compare models (only when needed)

You do NOT need this for the demo. Order matters, and each step needs the one before it.

```powershell
python -m src.features.build_table      # candidates + 27 features -> data\processed\*.parquet (takes a few minutes)
python -m src.models.run_baselines      # Logistic Regression + Random Forest
python -m src.models.run_baselines xgb  # XGBoost (default settings)
python -m src.models.run_decision_tree  # Decision Tree
python -m src.models.run_mlp            # MLP (shared pipeline)
python -m src.models.compare            # comparison table + figures -> results\tables, results\figures
python -m src.hmm.run_hmm_eval          # HMM/Viterbi decoder tuning + evaluation (uses the Random Forest)
```
`data\processed\` is not in git, so every teammate who wants to retrain runs `build_table` first.
Results can differ in the 3rd decimal between machines; quote one machine's numbers in the report.

**Which model the API serves:** the final model, `xgb_tuned` (Bayesian-tuned XGBoost + tuned HMM), is the default, so no
setting is needed. To demo another model (restart the backend):
```powershell
$env:PATHSNAP_MODEL = "rf"      # any name in models\ that also has models\hmm_params_<name>.json
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Models that have decoder settings: `rf`, `xgb_tuned`. For another model, first run
`python -m src.hmm.run_hmm_eval --model <name>`.

**Re-running the XGBoost tuning** (about 40 minutes; resumable): `python -m src.models.tune_xgb search --minutes 8`
(repeat until enough trials), then `python -m src.models.tune_xgb finalize`, then
`python -m src.hmm.run_hmm_eval --model xgb_tuned --trials 30`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named src...` | Run from the repo root, with the venv active, and use `python -m ...` |
| `No module named fastapi` (or similar) | `pip install -r requirements.txt` inside the active venv |
| Page says "Cannot reach the API" | Is the backend window running? Is the `?api=` address right (IP, port 8000, `http://`)? |
| Works on Laptop 1, not from Laptop 2 | Firewall rule (Part 3 step 2), same network, backend started with `--host 0.0.0.0`; try a phone hotspot |
| `/health` OK but trip list empty | `PATHSNAP_TRIPS_DIR` / `data\raw\student_test` is wrong or points one folder too high |
| `Feature table not found` | Run `python -m src.features.build_table` first (Part 4) |
| Port 8000 already in use | Close the old backend window, or use `--port 8001` and put 8001 in the UI address |
| Page loads but is blank / old | Hard refresh with `Ctrl+Shift+R` |
| Mock mode (no backend, for UI work) | `http://localhost:8080/?mock=1` |

## Where things are

| What | Where |
|---|---|
| Feature extraction | `src/features/candidates.py` |
| Models and evaluation | `src/models/` |
| HMM / Viterbi | `src/hmm/viterbi.py`, `src/inference/predict.py` |
| API | `api/main.py` (+ `api/service.py`, `api/scoring.py`); contract in `api/CONTRACT.md` |
| UI | `ui/index.html`, `ui/app.js` |
| Earlier MLP-only backend (kept, not used by the demo) | `api/legacy_main.py`, `src/hmm/legacy_viterbi.py`, `src/pipeline.py`, `ui/app.py` |
| Shared context for AI assistants | `AGENTS.md` |
