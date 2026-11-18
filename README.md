# PathSnap: map-matching and route reconstruction

Course project for UE24CS352A (PES University ML hackathon). Given a trip's noisy 1 Hz GPS points and its road
network, we predict which road arc the vehicle was on at each point and the resulting route.

Method: for each GPS point, the road arcs within 40 m become candidates. A classifier scores each (point, candidate)
pair using 27 features. An HMM decoded with Viterbi then picks a connected sequence of arcs. The final model is a tuned
XGBoost classifier plus the HMM.

## Results (student-test, 10 trips)

| | Top-1 | Macro F1 |
|---|---|---|
| Tuned XGBoost alone | 0.882 | 0.930 |
| Tuned XGBoost + HMM | 0.926 | 0.956 |

All five models and the cross-validated comparison are in `results/tables/model_comparison.md`. The tuning gain over
default XGBoost is small (+0.0003 CV macro F1, inside the fold-to-fold spread); the HMM gain is the large one.
Numbers move in the third decimal between machines.

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

On a Mac, XGBoost also needs `brew install libomp`.

The data is not in the repo. Put the provided `train/` and `student_test/` folders under `data/raw/`
(see `data/README.md`).

## Run the demo

Backend (serves the full pipeline: features, classifier, HMM):

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

UI (static files, in a second terminal or on a second laptop):

```bash
cd ui
python -m http.server 8080
```

Open http://localhost:8080 and point it at the API address. Endpoints are `GET /health`, `GET /trips` and
`POST /predict`; the contract is in `api/CONTRACT.md`. Accuracy is computed on the server and the reference route is
never sent to the UI. `PATHSNAP_MODEL` and `PATHSNAP_TRIPS_DIR` select the model and the trips folder.

Step-by-step instructions for one or two laptops are in `docs/SETUP_AND_RUN.md`.

## Reproduce the experiments

```bash
python -m src.features.build_table        # candidates + features
python -m src.models.run_baselines        # logistic regression, random forest
python -m src.models.run_baselines xgb    # XGBoost (default)
python -m src.models.run_decision_tree
python -m src.models.run_mlp
python -m src.models.tune_xgb search --minutes 8   # Bayesian optimisation, resumable
python -m src.models.tune_xgb finalize
python -m src.hmm.run_hmm_eval --model xgb_tuned --trials 30
python -m src.models.compare              # comparison table and figures
python -m src.models.confusion            # confusion matrices
python -m src.visualization.maps --partition student_test --trip 00000052
python reports/make_report_tables.py      # report tables from the logs
```

The EDA is `notebooks/01_eda.ipynb`.

## Layout

| Path | Contents |
|---|---|
| `src/data/` | loading and projection |
| `src/features/` | candidate generation, labelling, 27 features |
| `src/models/` | model runs, tuning, comparison, confusion matrices |
| `src/hmm/` | Viterbi decoder and its tuning |
| `src/inference/` | the prediction pipeline used by the API |
| `src/visualization/` | Folium maps |
| `api/`, `ui/` | backend and UI |
| `notebooks/` | EDA |
| `results/` | logs, tables, figures, maps |
| `reports/` | report template, table generator, generated report |
| `archive/` | earlier pipelines by two teammates that the final system does not use (see `archive/README.md`) |
| `docs/` | run guide |
| `AGENTS.md` | who did what and how AI tools were used |

## Notes

- The evaluator trips were never used for anything.
- Train trips 00000033 and 00000054 are excluded (under half their points could be labelled).
- The organisers' PDFs are not included in the repo.
