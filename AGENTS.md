# PathSnap: shared context for every teammate and every AI assistant

Read this fully before helping with anything in this repo. It is the single source of truth for what we are building and the decisions we have made. If something here conflicts with a chat suggestion, this file wins. If you think it is wrong, say so instead of silently deviating.

## 1. The task

PathSnap is a PES University ML hackathon (UE24CS352A). Given noisy 1 Hz GPS points from a vehicle and a road network, reconstruct the sequence of road segments (arcs) the vehicle drove.

Required pipeline:
1. Explore the raw data (EDA, Folium maps, data-quality log).
2. Candidate generation, labelling and feature engineering.
3. Train and compare **at least 5 models** from different families.
4. Pick the best by experimental evidence, then **Bayesian hyperparameter optimisation** (e.g. Optuna) with group-aware CV on the 80 training trips.
5. **HMM + Viterbi** route reconstruction on top of the classifier. Compare classifier-only vs classifier+HMM.
6. FastAPI backend (Laptop 1) and UI (Laptop 2) that lets an evaluator pick one of 10 hidden trips, runs the full pipeline, and shows the predicted route and accuracy.
7. Deliverables: public GitHub repo + one PDF technical report (structure in the organisers' Deliverables.pdf).

## 2. Data facts (verified by us, not assumed)

Layout: `data/raw/{train,student_test}/<trip_id>/<trip_id>.{track,nodes,arcs,route}`. All files are whitespace-separated, with no headers.
- `.track`: lon, lat, timestamp (s)
- `.nodes`: lon, lat (row index = node_id)
- `.arcs`: from_node, to_node (row index = arc_id), directed
- `.route`: reference arc_id sequence (the only ground truth)

Partitions: 80 train, 10 student_test. The 10 evaluator trips (IDs 15, 17, 23, 30, 49, 59, 67, 73, 87, 94 are missing from our folders) are held by the organisers.

Important properties:
- **Nodes and arcs are per-trip.** IDs are local to a trip. Never assume IDs mean the same thing across trips.
- **There are no per-point labels.** We derive them by projecting each GPS point onto the route arcs (nearest route arc, order-aware). Points more than ~30 m from the route are unreliable and should be dropped or flagged.
- **Trips are located all over the world** (trip centres range lon -100 to 137, lat -35 to 62), so no single CRS works. `src/data/loader.py` projects every trip to metres with its own local azimuthal-equidistant projection centred on the trip. Always use `x`/`y` from the loader. Never use Euclidean distance on degrees, and never hardcode a UTM zone. Models must not use absolute location as a feature (train and test trips are in different cities).
- Trip sizes are very uneven (361 to 19,313 points). Pooled metrics are dominated by big trips, so also report per-trip metrics.
- Sampling is only nominally 1 Hz (per trip 53% to 100% of steps are exactly 1 s).
- Disconnected road networks are normal (up to 421 components in a train trip). Duplicate arcs and self-loops exist.

Known problem trips (decide handling in the EDA, and log the decision):
- Train 54, 33, 02: the route does not match the track (route much shorter than track, large offsets). Labels unreliable; consider excluding or flagging.
- Large timestamp gaps: train 90 (153,064 s), 78 (52,810 s), 69, 21; student_test 65 (210 s).
- Speed teleports: train 79, 13, 33, 43; student_test 40.
- Routes with discontinuities (consecutive arcs not connected): 11 trips, including student_test 76.

## 3. Hard rules (academic integrity, from the organisers)

- **Evaluator-test data must never be used** for training, feature selection, tuning, thresholds, or model choice. It is not in this repo. Do not ask for it, simulate it, or hardcode anything about it.
- Ground-truth labels must stay server-side in the API and never reach the model or the UI.
- **No hardcoded predictions.** The API must run the real pipeline.
- Fit all learned preprocessing (scalers etc.) on training trips only.
- Student_test is for development comparison only. Do not tune on it so heavily that it becomes a training set.
- **Never fabricate** metrics, graphs, training curves, citations or logs. Every number in the report must come from a real run and be reproducible from this repo. Do not draw training curves for models that have no epoch history.
- Teams are compared for code and result similarity. Make your own choices and justify them from our own experiments.
- Every teammate must be able to explain any code, result or claim in the viva. If an AI writes something, the human must understand it before it is committed. Disclose AI assistance as the event requires.

## 4. Approach we agreed

**Candidate scoring, not direct classification.** For each GPS point, generate candidate arcs (all arcs within ~40 m, or the k nearest, via a KD-tree over arc points densified every ~10 m). Each (point, candidate) row gets a binary label: is this the true route arc for that point? The model outputs a probability per candidate. Observation-level Top-1 = the highest-probability candidate vs the true arc.

Candidate features (examples): point-to-arc distance, projection position along the arc, heading vs arc bearing, speed, time gap to the previous point, arc length, distance rank among candidates, neighbour-point context, connectivity to the previous best candidate.

Measured on our final labelling (train, kept trips): a 40 m radius misses about 0.2% of labelled points' true arcs, with about 6 candidates per point (see `notebooks/01_eda.ipynb`, section 7).

**The 5 models we trained (one family each), all on the shared table:**
1. Logistic Regression: linear/probabilistic baseline (`src/models/run_baselines.py`)
2. Decision Tree: single tree, depth chosen by a depth sweep (`src/models/run_decision_tree.py`)
3. Random Forest: bagging ensemble (`src/models/run_baselines.py`)
4. XGBoost: boosting. Selected as the final model on CV macro F1, then tuned with Optuna (`run_baselines.py xgb`, `tune_xgb.py`)
5. Shallow MLP: neural net, the only one with epoch-wise loss curves (`src/models/run_mlp.py`)

We did not build KNN or LightGBM. Final model: tuned XGBoost + HMM (`xgb_tuned`).

**HMM/Viterbi:** hidden states = candidate arcs at each time step. Emission = the classifier probability (optionally combined with a distance term). Transition = high for the same arc or a connected arc (to_node equals next from_node), decaying with graph distance, with a small floor for disconnected pairs so decoding can continue. Handle empty candidate sets and large time gaps explicitly (reset or widen). Tune decoder parameters on train + student_test only.

## 5. What every model must share (so the comparison is fair)

- **One shared feature table** built by one person. Everyone trains on the same table.
- **Fixed GroupKFold folds**, grouped by trip, created once with `SEED` from `src/config.py` and saved. All models use the same folds.
- **One shared evaluation function** (`evaluate()` in `src/models/common.py`) returning: macro F1, Top-1 accuracy, log loss, Candidate Miss Rate, route continuity, training time, inference time. Do not write your own metric code.
- **Common interface:** `fit(X, y)` and `predict_proba(X)`. Probabilities must be usable by the HMM.
- Results go to `results/logs/` in the same format for every model. Log the config, seed, and timings.
- Primary model-selection metric: CV macro F1 (also look at log loss and fold stability). Never select on training accuracy.

## 6. Who did what

| Person | Models | Other work |
|---|---|---|
| Niyatee | Logistic Regression, Random Forest, XGBoost tuning | shared pipeline (candidates, labels, features, folds, metrics), HMM/Viterbi, API, comparison, Folium maps, EDA notebook |
| Nishitha | Decision Tree | evaluator UI, route-continuity diagnostics |
| Abhijith | MLP | first MLP pipeline, loss curves |
| Nikhil | XGBoost (first version, hyper-parameters) | candidate and feature experiments |

Earlier individual pipelines are kept in `archive/` and are not used for reported results.

## 7. Repo conventions

- Layout: `src/{data,features,models,hmm,inference,visualization}`, `api/`, `ui/`, `notebooks/`, `models/` (saved weights), `results/{figures,maps,logs,tables}`, `reports/`, `tests/`, `archive/` (earlier individual pipelines, not used for results), `docs/`.
- Raw data is not committed (`data/raw/` is gitignored). Put the provided folders there locally.
- Shared constants (paths, seed, CRS) live in `src/config.py`. Loading goes through `src/data/loader.py`.
- Work on a branch per person or feature, open a PR into `main`, and keep commits small and descriptive.
- Never commit secrets, keys, or data files. Keep scripts reproducible: fixed seeds, no hand-edited outputs.
- Run `python -m pytest tests` before pushing.

## 8. How AI assistants should behave here

- Follow Sections 3 to 5. Ask before changing a shared decision (features, folds, metrics, interface).
- Do not touch other people's model files without being asked.
- Prefer small, explainable code over clever code. Add comments a teammate can use to explain it in a viva.
- When reporting results, say exactly which run and partition produced them. If something was not run, say it was not run.
- If a requested shortcut would break Section 3 (leakage, hardcoding, fabricated results), refuse and say why.
