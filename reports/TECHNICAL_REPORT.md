# PathSnap: Map-Matching and Route Reconstruction

**Technical report. UE24CS352A, PES University.**

Team: TODO (names and SRNs). Repository: https://github.com/NIYATEE29/ML_Hackathon

> **How to use this file.** It is a template. Text you write goes here (`reports/report_template.md`). Run
> `python reports/make_report_tables.py` to fill every `{{...}}` table from the committed logs and write
> `reports/TECHNICAL_REPORT.md`, which you convert to PDF. Do not edit `TECHNICAL_REPORT.md` directly, because it is
> overwritten. Lines marked **TODO** are questions the organisers require us to answer from our own results and
> understanding. Each of us should be able to explain every sentence in the viva. Delete this note before exporting.

Everything here can be reproduced from the repository: see section 7 for the commands and `docs/SETUP_AND_RUN.md`.

---

## 2.2 Dataset understanding and exploratory data analysis

Partitions: `train` (80 trips) for fitting, cross-validation and tuning; `student_test` (10 trips) for development
comparison only; the evaluator trips are never available to us and were never used. Each analysis below says which
partition it uses. The full analysis is in `notebooks/01_eda.ipynb`; the figures are shown in its outputs.

### Preprocessing

(Facts about what the code does. Rewrite in your own words.)

- Each trip is projected to metres with its own local azimuthal-equidistant projection, because the trips are spread over the whole world (`src/data/loader.py`).
- For every GPS point the road arcs within 40 m (at most the 12 nearest) are candidates. They come from a KD-tree built over the arcs sampled every 10 m (`src/features/candidates.py`).
- There are no per-point labels, so we derive them. Each point is matched to the reference route with a forward-moving pointer (window of 60 route arcs, 3 arcs of backward slack, 30 m maximum distance, re-anchoring after 20 unmatched points).
- Trips where fewer than half the points could be labelled are excluded from training and evaluation: train trip 00000033 (29% labelled), train trip 00000054 (11% labelled).
- Cross-validation is 5-fold, grouped by trip, with the folds created once and saved (`results/folds.json`).

### Missing values and data quality (`train` and `student_test`)

| partition | trips | gps_points | min_points | max_points | trips_with_gap_over_60s | trips_with_route_breaks |
|---|---|---|---|---|---|---|
| student_test | 10 | 19160 | 262 | 3958 | 1 | 1 |
| train | 80 | 208417 | 361 | 19313 | 22 | 10 |

Details per trip: `results/tables/data_quality.csv`. Summary of the missing-value check: see notebook section 1.

**TODO (team):** state in your own words what the missing-value analysis found and what you did about the problems it did find (time gaps, GPS jumps, fragmented networks, routes that do not match the track).

### Figures to include

The EDA figures are drawn inside `notebooks/01_eda.ipynb` (run it, then right-click or save each figure from the output). Pick the ones you discuss:
trip sizes and geography, sampling and time gaps, speeds, trajectories over the road network, network structure, reference routes, labelling and the candidate radius, feature distributions and correlation, feature importance. Folium map: see section 6.

### Patterns observed and features

**TODO (team):** what patterns did you see, and why did you choose the 27 features? The notebook lists the six feature groups (distance, position on the arc, direction, motion, topology, neighbour context) with a short reason for each, in section 8.

---

## 3. Model experimentation and comparison

Five models were trained and compared on the same candidate table, the same folds and the same evaluation function
(`src/models/common.py`): Logistic Regression, Decision Tree, Random Forest, XGBoost and an MLP. The primary selection metric is
cross-validated macro F1 of the per-point top-1 choice; Top-1 accuracy, log loss, route continuity and inference time are reported too.

### 3.1 Overall comparison

| Model | Algorithm Family | Key Configuration | CV Macro F1 (mean) | CV Macro F1 (std) | CV Top-1 | CV Log Loss | Student Top-1 | Route Continuity | Route Continuity (transitions only) | Student Macro F1 | Student Log Loss | Train Time (s) | Inference (ms/point) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | C=1.0, max_iter=1000, solver=lbfgs | 0.9325 | 0.0078 | 0.8859 | 0.1357 | 0.8533 | 0.9265 | nan | 0.9138 | 0.1428 | 2.0206 | 0.0010 |
| Decision Tree | Decision tree (single, non-ensemble) | max_depth=10, min_samples_leaf=20 | 0.9378 | 0.0076 | 0.8947 | 0.1095 | 0.8757 | 0.9371 | 0.7548 | 0.9269 | 0.1071 | 12.3019 | 0.0002 |
| Random Forest | Bagging ensemble (trees) | n_estimators=100, max_depth=None, min_samples_leaf=20, max_features=sqrt, max_samples=0.25 | 0.9404 | 0.0074 | 0.8991 | 0.0945 | 0.8740 | 0.9389 | nan | 0.9259 | 0.0993 | 70.2701 | 0.0295 |
| XGBoost | Gradient boosting (trees) | n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, tree_method=hist | 0.9418 | 0.0076 | 0.9015 | 0.0912 | 0.8841 | 0.9457 | 0.7769 | 0.9319 | 0.0941 | 18.2317 | 0.0147 |
| XGBoost (tuned) | Gradient boosting (trees), Bayesian-tuned | n_estimators=200, max_depth=10, learning_rate=0.03699197382485672, subsample=0.8780911808604259, colsample_bytree=0.6387322148808915, min_child_weight=1.2469770363038049, reg_lambda=7.501182296073602, reg_alpha=0.2481316839035101, gamma=3.740093841528239, tree_method=hist | 0.9421 | 0.0075 | 0.9020 | 0.0906 | 0.8817 | 0.9462 | 0.7809 | 0.9304 | 0.0943 | 16.9285 | 0.0164 |
| MLP | Neural network (MLP) | hidden_layer_sizes=[64, 32], activation=relu, solver=adam, alpha=0.0001, learning_rate_init=0.001, batch_size=1024, epochs=20 | 0.9374 | 0.0074 | 0.8941 | 0.1038 | 0.8809 | 0.9431 | 0.7684 | 0.9300 | 0.1052 | 25.2794 | 0.0021 |

Figures: `results/figures/cmp_cv_macro_f1.png`, `results/figures/cmp_student_top1_f1.png`, `results/figures/cmp_log_loss.png`, `results/figures/cmp_f1_vs_inference.png`.

**TODO (team):** how did you choose the final model, and how big were the differences compared with the fold-to-fold variation?

### Evidence used in every model subsection (train vs student-test)

| Model | CV macro F1 | CV std | Train Top-1 | Student Top-1 | Train macro F1 | Student macro F1 | Student log loss | Fit time (s) | Inference (ms/point) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9325 | 0.0078 | 0.8867 | 0.8533 | 0.9329 | 0.9138 | 0.1428 | 2.0206 | 0.0010 |
| Decision Tree | 0.9378 | 0.0076 | 0.9096 | 0.8757 | 0.9466 | 0.9269 | 0.1071 | 12.3019 | 0.0002 |
| Random Forest | 0.9404 | 0.0074 | 0.9102 | 0.8740 | 0.9469 | 0.9259 | 0.0993 | 70.2701 | 0.0295 |
| XGBoost (default settings) | 0.9418 | 0.0076 | 0.9216 | 0.8841 | 0.9538 | 0.9319 | 0.0941 | 18.2317 | 0.0147 |
| MLP | 0.9374 | 0.0074 | 0.9116 | 0.8809 | 0.9478 | 0.9300 | 0.1052 | 25.2794 | 0.0021 |
| XGBoost (tuned) | 0.9421 | 0.0075 | 0.9282 | 0.8817 | 0.9577 | 0.9304 | 0.0943 | 16.9285 | 0.0164 |

### Configuration and optimisation objective per model

| Model | Hyper-parameters (initial / final values) | Optimisation objective |
|---|---|---|
| Logistic Regression | {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"} | Log loss (binary cross-entropy) with an L2 penalty on the weights |
| Decision Tree | {"max_depth": 10, "min_samples_leaf": 20} | Gini impurity, minimised greedily at each split |
| Random Forest | {"n_estimators": 100, "max_depth": null, "min_samples_leaf": 20, "max_features": "sqrt", "max_samples": 0.25} | Gini impurity in each tree; trees are fitted on bootstrap samples and averaged |
| XGBoost (default settings) | {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0, "tree_method": "hist"} | Logistic loss with L2 (lambda) and L1 (alpha) penalties on leaf weights and a minimum split gain (gamma) |
| MLP | {"hidden_layer_sizes": [64, 32], "activation": "relu", "solver": "adam", "alpha": 0.0001, "learning_rate_init": 0.001, "batch_size": 1024, "epochs": 20} | Log loss (binary cross-entropy) minimised with Adam and an L2 penalty (alpha) |
| XGBoost (tuned) | {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.03699197382485672, "subsample": 0.8780911808604259, "colsample_bytree": 0.6387322148808915, "min_child_weight": 1.2469770363038049, "reg_lambda": 7.501182296073602, "reg_alpha": 0.2481316839035101, "gamma": 3.740093841528239, "tree_method": "hist"} | Same as XGBoost: logistic loss with L2/L1 penalties and a minimum split gain |

### Macro F1 on each CV fold

| Model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9220 | 0.9312 | 0.9335 | 0.9320 | 0.9439 |
| Decision Tree | 0.9282 | 0.9353 | 0.9416 | 0.9353 | 0.9485 |
| Random Forest | 0.9302 | 0.9382 | 0.9425 | 0.9403 | 0.9506 |
| XGBoost (default settings) | 0.9307 | 0.9395 | 0.9447 | 0.9427 | 0.9514 |
| MLP | 0.9274 | 0.9360 | 0.9420 | 0.9347 | 0.9468 |
| XGBoost (tuned) | 0.9306 | 0.9411 | 0.9448 | 0.9425 | 0.9515 |

### 3.2 Per-model subsections

For each model answer A (understanding), B (training and configuration) and C (results). Use the tables above and the
logs in `results/logs/`. Do not write textbook descriptions; tie each answer to what we observed.

#### Logistic Regression

Code: `src/models/run_baselines.py logreg`. Features are standardised inside the pipeline (fitted on training folds only).
- **A1.** TODO why suitable. **A2.** TODO strengths and limitations here.
- **B5.** Initial hyper-parameters: see the configuration table. **B6.** Objective: see the table.
- **C7, C8.** Student-test results: see the evidence table. **C9.** TODO over/under-fitting and the evidence (compare train and student columns).

#### Decision Tree

Code: `src/models/run_decision_tree.py`. Depth chosen by a sweep: `results/logs/dtree_depth_sweep.csv`, `results/figures/dtree_depth.png`; feature importance: `results/tables/dtree_feature_importance.csv`.
- **A1, A2.** TODO. **B5, B6.** See the tables; TODO explain how the depth was chosen. **C7 to C9.** TODO.

#### Random Forest

Code: `src/models/run_baselines.py rf`.
- **A1, A2.** TODO. **B5, B6.** See the tables; TODO explain why each tree is fitted on a quarter of the rows (`max_samples`). **C7 to C9.** TODO.

#### XGBoost

Code: `src/models/run_baselines.py xgb` (default settings) and `src/models/tune_xgb.py` (tuned, section 4).
- **A1, A2.** TODO. **B5, B6.** See the tables. **C7 to C9.** TODO.

#### MLP

Code: `src/models/run_mlp.py`. This is the only model with an epoch history, so it is the only one with training and validation loss curves: `results/figures/mlp_loss_curves_shared.png`, data in `results/logs/mlp_epoch_history_shared.csv`.
- **A1, A2.** TODO. **B5, B6.** See the tables. **C7 to C9.** TODO: what do the loss curves show about over/under-fitting?

### 3.3 Visualisations

- CV macro F1, student Top-1 and macro F1, log loss and speed versus accuracy: the `cmp_*.png` figures above.
- Training and validation loss: MLP only (above). No curves were drawn for the other models because they have no epoch history.
- Confusion matrices (diagnostic): `results/figures/confusion_matrices.png`, code `src/models/confusion.py`. A row is a (GPS point, candidate arc) pair; "chosen" means the highest-probability candidate of its point. The last panel is the final model with the HMM deciding the choice.

| model | tn | fp | fn | tp | true_arc_recall | true_arc_precision | macro_f1_from_matrix | macro_f1_logged |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 103308 | 2759 | 2751 | 16051 | 0.8537 | 0.8533 | 0.9138 | 0.9138 |
| Decision Tree | 103701 | 2366 | 2358 | 16444 | 0.8746 | 0.8742 | 0.9261 | 0.9269 |
| Random Forest | 103696 | 2371 | 2363 | 16439 | 0.8743 | 0.8740 | 0.9259 | 0.9259 |
| XGBoost | 103886 | 2181 | 2173 | 16629 | 0.8844 | 0.8841 | 0.9319 | 0.9319 |
| XGBoost (tuned) | 103841 | 2226 | 2218 | 16584 | 0.8820 | 0.8817 | 0.9304 | 0.9304 |
| MLP | 103827 | 2240 | 2232 | 16570 | 0.8813 | 0.8809 | 0.9300 | 0.9300 |
| XGBoost (tuned) + HMM | 104641 | 1426 | 1418 | 17384 | 0.9246 | 0.9242 | 0.9555 | 0.9563 |

- Folium maps: section 6.

---

## 4. Bayesian hyper-parameter optimisation (final model: XGBoost)

Code: `src/models/tune_xgb.py`. Logs: `results/logs/xgb_tuning_trials.csv`, `results/logs/xgb_best_params.json`. Figures: `results/figures/xgb_tuning_history.png`, `results/figures/xgb_tuning_importance.png`.

Optuna TPE sampler, seed 42, objective = mean macro F1 over the 5 trip-grouped CV folds on the training trips (maximised). Trials run: 42 (13 completed, 29 stopped early by the median pruner). Trial 0 is the untuned default.

**Hyper-parameters and ranges searched** (Q1, Q2):

| Hyper-parameter | Range searched |
|---|---|
| n_estimators | integer 100 to 500, step 50 |
| max_depth | integer 3 to 10 |
| learning_rate | 0.02 to 0.3 (log scale) |
| subsample | 0.5 to 1.0 |
| colsample_bytree | 0.4 to 1.0 |
| min_child_weight | 1.0 to 50.0 (log scale) |
| reg_lambda | 0.01 to 30.0 (log scale) |
| reg_alpha | 0.001 to 10.0 (log scale) |
| gamma | 0.0 to 5.0 |

**Objective** (Q3): mean cross-validated macro F1 on the five fixed trip-grouped folds, training trips only. Student-test was not used during the search.

**Best values** (Q5):

```json
{
 "n_estimators": 200,
 "max_depth": 10,
 "learning_rate": 0.03699197382485672,
 "subsample": 0.8780911808604259,
 "colsample_bytree": 0.6387322148808915,
 "min_child_weight": 1.2469770363038049,
 "reg_lambda": 7.501182296073602,
 "reg_alpha": 0.2481316839035101,
 "gamma": 3.740093841528239,
 "tree_method": "hist"
}
```

**Untuned versus tuned** (Q6), same folds and partitions:

| Model | CV macro F1 | Student Top-1 | Student macro F1 | Student log loss | Inference (ms/point) |
|---|---|---|---|---|---|
| XGBoost (untuned) | 0.9418 | 0.8841 | 0.9319 | 0.0941 | 0.0147 |
| XGBoost (tuned) | 0.9421 | 0.8817 | 0.9304 | 0.0943 | 0.0164 |

Tuned minus untuned CV macro F1 per fold: -0.0001, +0.0015, +0.0001, -0.0002, +0.0000 (tuned better on 3 of 5 folds; fold standard deviation is 0.0075).

**TODO (team):** did the tuned model outperform the untuned one, and by how much compared with the fold variation? Which model do we use in the demo, and why?

---

## 5. HMM and Viterbi route reconstruction

Code: `src/hmm/viterbi.py` (decoder), `src/hmm/run_hmm_eval.py` (tuning and evaluation), `src/inference/predict.py`.

Decoder parameters were tuned on out-of-fold training probabilities (30 Optuna trials on 20 training trips, objective: decoded Top-1 on train OOF) and then evaluated once on student-test.

| Method | Top-1 | Macro F1 | Route continuity | Continuity (real arc changes only) | Route F1 |
|---|---|---|---|---|---|
| Classifier only (XGBoost tuned) | 0.8817 | 0.9304 | 0.9448 | 0.7785 | 0.8942 |
| Classifier + HMM (default decoder settings) | 0.9055 | 0.9445 | 0.9704 | 0.8403 | 0.9192 |
| Classifier + HMM (tuned decoder settings) | 0.9256 | 0.9563 | 0.9836 | 0.8986 | 0.9340 |

Tuned decoder parameters:

```json
{
 "emission_weight": 0.33947335663976347,
 "p_floor": 0.0001,
 "switch_penalty": 1.9728163193628445,
 "hop_penalty": 1.3111615934447882,
 "reverse_penalty": 6.700023135541605,
 "disconnected_penalty": 12.58847978307178,
 "max_hops": 4,
 "hops_per_sec": 1.1646215279610348,
 "gap_reset_s": 30.0
}
```

Per student-test trip:

| Trip | Points | Top-1 classifier | Top-1 HMM | Top-1 change | Route F1 classifier | Route F1 HMM | Route F1 change |
|---|---|---|---|---|---|---|---|
| 00000000 | 2501 | 0.9588 | 0.9664 | 0.0076 | 0.9149 | 0.9494 | 0.0345 |
| 00000007 | 1327 | 0.9631 | 0.9985 | 0.0354 | 0.9758 | 0.9878 | 0.0119 |
| 00000009 | 1021 | 0.9050 | 0.9510 | 0.0460 | 0.8606 | 0.8965 | 0.0359 |
| 00000040 | 2946 | 0.9599 | 0.9705 | 0.0105 | 0.9444 | 0.9786 | 0.0342 |
| 00000052 | 262 | 0.9733 | 0.9962 | 0.0229 | 0.8778 | 0.9266 | 0.0488 |
| 00000056 | 3761 | 0.8585 | 0.8942 | 0.0356 | 0.9070 | 0.9264 | 0.0195 |
| 00000065 | 768 | 0.8242 | 0.8294 | 0.0052 | 0.8904 | 0.9386 | 0.0482 |
| 00000076 | 1485 | 0.9347 | 0.9724 | 0.0377 | 0.8647 | 0.9174 | 0.0527 |
| 00000077 | 3865 | 0.7397 | 0.8595 | 0.1198 | 0.8648 | 0.9679 | 0.1031 |
| 00000091 | 874 | 0.9062 | 0.9291 | 0.0229 | 0.8414 | 0.8509 | 0.0095 |

**TODO (team), answer the six questions from our code and results:**
1. Hidden states: read `src/hmm/viterbi.py`.
2. Emission probabilities: how the classifier probability enters (see `emission_weight`).
3. Transition probabilities: how graph hops, reversals and disconnected pairs are scored (`hop_penalty`, `reverse_penalty`, `disconnected_penalty`, `max_hops`).
4. Missing candidates and large time gaps (`gap_reset_s`, `p_floor`).
5. Disconnected road segments.
6. Did decoding help? Use the tables above. Also discuss where it helped least (look at the per-trip table), and note that the reference route itself has breaks (`route_breaks` in `data_quality.csv`), so continuity numbers must be read with care.

---

## 6. Geographical visualisation with Folium

Code: `src/visualization/maps.py`. Maps: `results/maps/student_test_00000052_prediction.html`, `student_test_00000065_prediction.html`, `student_test_00000077_prediction.html`. They are produced by running our real pipeline (classifier, then HMM) on the trip.

Layers: GPS observations (blue), road network near the track (grey), reference route (green), predicted route (red), classifier-only route (orange, dashed), points where the HMM changed the classifier's choice (purple). There is a legend, a layer control, and the view is fitted to the trip.

**TODO (team):** insert at least one screenshot of a map (open the HTML in a browser with internet, so the basemap loads) and describe what it shows. Trip 52 is small and clean, trip 65 has a 210 s time gap, trip 77 is the hardest.

---

## 7. API, UI and deployment

- Backend (Laptop 1): FastAPI, `api/main.py`, `api/service.py`, `api/scoring.py`; contract in `api/CONTRACT.md`. Endpoints: `GET /health`, `GET /trips`, `POST /predict`. The API runs the full pipeline (features, classifier, HMM). Accuracy is computed on the server and the reference route is never sent to the UI (`tests/test_api.py` checks this).
- UI (Laptop 2): `ui/` (plain HTML, CSS and JavaScript with Leaflet, no build step).
- Run commands: see `README.md` and `docs/SETUP_AND_RUN.md`.

**TODO (team):** insert screenshots of the working UI and of successful API responses in the console of the receiving laptop.

---

## 8. Academic integrity and AI assistance

**TODO (team):** write the AI-assistance disclosure required by the event.
