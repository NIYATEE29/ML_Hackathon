# PathSnap: Map-Matching & Route Reconstruction (ML Hackathon)

Reconstruct a vehicle's road-segment route from noisy GPS observations using a
candidate-scoring classifier plus HMM/Viterbi decoding, served through a FastAPI
backend and a UI.

## Repository layout

| Folder | Purpose |
|---|---|
| `data/` | Raw and processed data (not committed; see `data/README.md`) |
| `src/data/` | Raw-data loading, cleaning, preprocessing |
| `src/features/` | Candidate generation, labelling and feature engineering |
| `src/models/` | Model training, evaluation, Bayesian optimisation |
| `src/hmm/` | HMM and Viterbi route reconstruction |
| `notebooks/` | EDA and experiment notebooks |
| `models/` | Saved model weights and configs |
| `api/` | FastAPI backend |
| `ui/` | Evaluator-facing UI |
| `results/` | Figures, Folium maps, experiment logs and tables |
| `reports/` | Technical report |
| `tests/` | Unit tests |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the provided data under `data/raw/` (see `data/README.md`), then run tests:

```bash
python -m pytest tests
```

## Data partitions

- `train` (80 trips): fitting, group cross-validation, optimisation.
- `student_test` (10 trips): development comparison.
- Evaluator-test trips are never used during development.

## Status

- [x] Project scaffold
- [ ] Dataset exploration
- [ ] Candidate generation & features
- [ ] Model comparison (>= 5 models)
- [ ] Bayesian optimisation
- [ ] HMM / Viterbi
- [ ] API & UI
- [ ] Report
