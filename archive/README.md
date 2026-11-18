# Archive: earlier individual pipelines

Each teammate first built a model end to end on their own. When we compared models we moved to one shared
pipeline (one candidate table, one set of folds, one evaluation function) so the numbers are comparable.
Everything the final results come from is outside this folder. What is here was **not** used for any number in
the report; it is kept so the individual work is visible.

| Folder | What it is |
|---|---|
| `abhijith_mlp/` | First MLP pipeline: 9-feature extraction, its own folds and evaluation, HMM decoder, FastAPI app and Streamlit UI, and the first report draft. The MLP in the final comparison is `src/models/run_mlp.py`, which uses the same MLP settings on the shared table. |
| `nikhil_xgboost/` | First XGBoost pipeline: candidate generation, features, labels and training scripts. The final XGBoost is `src/models/run_baselines.py xgb`, using these hyper-parameters on the shared table. |
| `artifacts/` | Files these pipelines produced (saved MLP, logs, maps, loss curves). |
| `legacy_config.py`, `legacy_loader.py` | Settings and a loader helper that only these pipelines use. |

Why they are not the final pipeline: they use different features and labels from each other, so their scores
cannot be compared with each other or with the shared pipeline. For example the first MLP reports a 0.755
cross-validated Top-1 on its own table, which is not comparable to the shared-table numbers.

Run their tests with `python -m pytest archive` (extra packages: `pip install -r archive/requirements.txt`).
The archived API had a `/map` endpoint that drew the ground-truth route; it was removed because ground truth must
never reach a client.
