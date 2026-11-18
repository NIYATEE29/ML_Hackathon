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
- Trips where fewer than half the points could be labelled are excluded from training and evaluation: {{excluded}}.
- Cross-validation is 5-fold, grouped by trip, with the folds created once and saved (`results/folds.json`).

### Missing values and data quality (`train` and `student_test`)

{{datafacts}}

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

{{overall}}

Figures: `results/figures/cmp_cv_macro_f1.png`, `results/figures/cmp_student_top1_f1.png`, `results/figures/cmp_log_loss.png`, `results/figures/cmp_f1_vs_inference.png`.

**TODO (team):** how did you choose the final model, and how big were the differences compared with the fold-to-fold variation?

### Evidence used in every model subsection (train vs student-test)

{{evidence}}

### Configuration and optimisation objective per model

{{config}}

### Macro F1 on each CV fold

{{folds}}

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

{{confusion}}

- Folium maps: section 6.

---

## 4. Bayesian hyper-parameter optimisation (final model: XGBoost)

Code: `src/models/tune_xgb.py`. Logs: `results/logs/xgb_tuning_trials.csv`, `results/logs/xgb_best_params.json`. Figures: `results/figures/xgb_tuning_history.png`, `results/figures/xgb_tuning_importance.png`.

{{tuning_intro}}

**Hyper-parameters and ranges searched** (Q1, Q2):

{{tuning_space}}

**Objective** (Q3): mean cross-validated macro F1 on the five fixed trip-grouped folds, training trips only. Student-test was not used during the search.

**Best values** (Q5):

{{tuning_best}}

**Untuned versus tuned** (Q6), same folds and partitions:

{{tuning_cmp}}

{{tuning_perfold}}

**TODO (team):** did the tuned model outperform the untuned one, and by how much compared with the fold variation? Which model do we use in the demo, and why?

---

## 5. HMM and Viterbi route reconstruction

Code: `src/hmm/viterbi.py` (decoder), `src/hmm/run_hmm_eval.py` (tuning and evaluation), `src/inference/predict.py`.

{{hmm_intro}}

{{hmm_table}}

Tuned decoder parameters:

{{hmm_params}}

Per student-test trip:

{{hmm_pertrip}}

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
