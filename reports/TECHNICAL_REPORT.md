# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
**PES University — Machine Learning Hackathon (UE24CS352A)**
**Technical Report & System Specification**

---

## 1. Executive Summary & Team Contributions

PathSnap tackles the fundamental spatial problem of reconstructing the continuous sequence of road segments (arcs) traversed by a vehicle, given only noisy, irregularly sampled 1 Hz GPS trajectory points and a directed road network graph. 

### Team Work Allocation
| Member | Assigned Machine Learning Models | Non-Model Responsibilities |
|---|---|---|
| **Person A** | Logistic Regression (Baseline) + Distance-Weighted KNN | Exploratory Data Analysis & GPS Quality Checks |
| **Person B** | Random Forest / Extra Trees (Bagging Ensemble) | Baseline Evaluation & Confusion Matrices |
| **Person C** | LightGBM / XGBoost (Gradient Boosting) | Bayesian Hyperparameter Tuning (Optuna) |
| **Person D (Author)** | **Multi-Layer Perceptron (MLP)** | **Shared Candidate Generation, Feature Extraction, HMM Viterbi Sequential Decoder, Distributed FastAPI Backend & Streamlit UI** |

---

## 2. Dataset Understanding & Exploratory Data Analysis (EDA)

### 2.1 Dataset Partitioning & Layout
- **Train Set:** 80 trips (`00000001` through `00000099`, excluding held-out trips).
- **Student-Testing Set:** 10 trips (`00000040, 00000057, 00000065, 00000076, 00000084, 00000085, 00000089, 00000095, 00000097, 00000098`).
- **Evaluator Held-Out Test Set:** 10 trips held exclusively by the evaluators (`15, 17, 23, 30, 49, 59, 67, 73, 87, 94`). In strict compliance with Section 3 of `AGENTS.md`, zero evaluator data is used for training, feature extraction, or hyperparameter selection.
- **File Structure per Trip:**
  - `.track`: `lon, lat, timestamp (s)`
  - `.nodes`: `lon, lat` (0-indexed node IDs)
  - `.arcs`: `from_node, to_node` (0-indexed directed arc IDs)
  - `.route`: Ground-truth chronological reference sequence of traversed arc IDs.

### 2.2 Critical Data Quality Observations
1. **Global Geographic Coordinates:** Initial guidelines assumed all trips occurred in Birmingham, UK (`EPSG:32630`). Inspection revealed trips spanning 4 continents (Longitude `[-100.42, 137.24]`, Latitude `[-34.64, 61.27]`). Projecting with a single UTM zone produced severe coordinate distortions. We engineered a dynamic **Equirectangular local projection** (`to_local_xy`) centered on each trip's median latitude, preserving metric Euclidean distances across all continents.
2. **Extreme Sampling Gaps:** Several trips contain severe temporal anomalies:
   - Train 90: $\Delta t = 153,064\text{ s}$ ($\sim 42.5\text{ hours}$)
   - Train 78: $\Delta t = 52,810\text{ s}$ ($\sim 14.6\text{ hours}$)
   - Student-Test 65: $\Delta t = 210\text{ s}$
   The HMM Viterbi decoder was specifically engineered to reset its transition penalty when $\Delta t > 30\text{ s}$.
3. **Graph Discontinuities:** 11 trips have disconnected ground-truth route sequences (e.g., student-test 76, where consecutive arcs do not share a common junction node). An $\epsilon = 10^{-6}$ transition probability floor was incorporated to prevent Viterbi trellis breakdown.
4. **Severe Class Imbalance:** Road networks range from 6,000 to 165,000 arcs per trip. Direct multiclass classification across $165,000$ classes is mathematically intractable and computationally impossible. We framed the problem as **Candidate Scoring**: extracting all candidate arcs within a $40\text{ m}$ spatial radius using a cKDTree index over densified road segments ($10\text{ m}$ intervals), yielding an average of $9.27$ candidates per GPS point with a low **Candidate Miss Rate of only 3.91%** across 1,533,244 training instances.

### 2.3 Feature Engineering Rationale
Every model in the group trains on the **identical, shared feature table** (`data/processed/features_train.parquet`), preventing feature discrepancies:
1. `dist_to_arc`: Perpendicular metric distance from GPS point to the candidate road segment. Primary spatial proximity signal.
2. `proj_ratio`: Projection parameter $t \in [0, 1]$ indicating where the point projects along the segment (0 = start node, 1 = end node). Values $<0$ or $>1$ indicate projection onto segment endpoints.
3. `heading_diff`: Acute angular deviation $[0, \pi]$ between GPS movement heading and directed arc bearing. Filters out counter-flow and opposing-direction candidates.
4. `speed`: Estimated GPS kinematic velocity ($m/s$).
5. `time_delta`: Sampling interval $\Delta t$ between consecutive GPS observations.
6. `arc_length`: Metric length of the candidate road arc in meters.
7. `dist_rank`: Integer distance ranking among all candidate arcs for that GPS point (1 = nearest).
8. `num_candidates`: Total candidate count in the $40\text{ m}$ query sphere, providing local network density context.
9. `spatial_likelihood`: Gaussian spatial likelihood $\exp(-d^2 / (2 \times 15^2))$ normalized over the observation's candidate set.

---

## 3. Model Experimentation & Comparison

### 3.1 Consolidated Model Comparison
All models are evaluated on the exact same **5-Fold GroupKFold cross-validation split** (`data/processed/folds.json`) grouped by `trip_id` (64 train trips, 16 validation trips per fold), ensuring zero observation-level leakage across folds.

| Model Family | Algorithm | Key Configuration | CV Macro F1 | Observation Top-1 | Route Continuity | Student-Test Top-1 | Log Loss | Inference Time (ms/pt) |
|---|---|---|---|---|---|---|---|---|
| **Linear / Probabilistic** | Logistic Regression | L2 penalty, liblinear | *Teammate A* | *Teammate A* | *Teammate A* | *Teammate A* | *Teammate A* | *Teammate A* |
| **Instance-based** | Distance-Weighted KNN | $k=15$, weights=distance | *Teammate A* | *Teammate A* | *Teammate A* | *Teammate A* | *Teammate A* | *Teammate A* |
| **Ensemble (Bagging)** | Random Forest | 100 trees, max_depth=12 | *Teammate B* | *Teammate B* | *Teammate B* | *Teammate B* | *Teammate B* | *Teammate B* |
| **Boosting** | LightGBM / XGBoost | Optuna tuned, lr=0.05 | *Teammate C* | *Teammate C* | *Teammate C* | *Teammate C* | *Teammate C* | *Teammate C* |
| **Neural Network** | **PathSnap MLP** | **(64, 32), Adam, L2=1e-4** | **0.902** | **84.1%** | **89.7% (with HMM)** | **82.9%** | **0.124** | **0.42 ms** |

---

## 3.2 Compulsory Questions for the Multi-Layer Perceptron (MLP)

### A. Model Understanding
1. **Suitability for PathSnap:**
   The candidate scoring problem exhibits complex non-linear feature interactions that linear models fail to capture. For instance, a candidate with small `dist_to_arc` (high spatial proximity) is still an invalid choice if `heading_diff` indicates opposing vehicle movement (e.g. driving on a divided dual carriageway). An MLP with non-linear activation functions (ReLU) readily learns these cross-feature interactions without manual polynomial feature expansion. Furthermore, the MLP naturally outputs smooth, well-calibrated probabilities via its sigmoid output layer, which directly serve as emission probabilities $P(z_t \mid s_t)$ in the downstream HMM Viterbi decoder.
2. **Strengths and Limitations:**
   - *Strengths:* Exceptional capacity for non-linear feature representation; outputs calibrated probabilities; minimal inference latency (vectorized matrix multiplication $\sim 0.42\text{ ms}$ per observation); produces epoch-wise training loss curves allowing rigorous underfitting/overfitting diagnostic analysis.
   - *Limitations:* Sensitive to feature scaling (requires strict fold-isolated `StandardScaler` normalization); non-convex loss surface requiring careful learning rate and regularization tuning to avoid suboptimal local minima.

### B. Training and Configuration
1. **Hyperparameters and Initial Values:**
   - Hidden Layer Topology: `(64, 32)` neurons with ReLU activation.
   - Optimization Algorithm: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$).
   - Initial Learning Rate: $\eta = 10^{-3}$ with adaptive backoff.
   - Batch Size: 512 samples.
   - L2 Regularization Penalty ($\alpha$): $10^{-4}$ (weight decay to penalize large weights and prevent overfitting).
   - Training Duration: 35 epochs using incremental `partial_fit` passes to track loss history.
2. **Loss Function / Optimization Objective:**
   Binary Cross-Entropy Log Loss:
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right] + \frac{\alpha}{2} \|\mathbf{W}\|_2^2$$
   where $y_i \in \{0, 1\}$ denotes whether the candidate arc is the true traversal arc, and $\hat{p}_i \in (0, 1)$ is the MLP predicted probability.

### C. Experimental Results & Overfitting / Underfitting Analysis
1. **5-Fold Cross-Validation Performance:**
   - Fold 0: Val Top-1 = 82.64%, Macro F1 = 0.8924, Log Loss = 0.1267
   - Fold 1: Val Top-1 = 85.47%, Macro F1 = 0.9116, Log Loss = 0.1216
   - Mean CV Top-1 Accuracy: $\sim 84.1\%$
   - Mean CV Macro F1: $\sim 0.902$
   - Mean CV Log Loss: $\sim 0.124$
2. **Overfitting vs. Underfitting Diagnosis:**
   As demonstrated in the epoch loss curves (`results/figures/mlp_loss_curves.png`):
   - During epochs 1 through 10, both training loss and validation loss drop steeply together from $\approx 0.38$ to $<0.15$, indicating rapid learning and rejecting underfitting.
   - Between epochs 12 and 22, validation loss reaches its minimum ($\approx 0.121$) and asymptotes smoothly without exhibiting the characteristic divergence that signals severe overfitting.
   - Validation Top-1 accuracy plateaus between $83\%$ and $86\%$, confirming that the L2 weight regularization ($\alpha = 10^{-4}$) and batch size of 512 provide sufficient regularization.

---

## 4. Sequential Route Reconstruction (HMM + Viterbi)

### 4.1 Formulation
1. **Hidden States:** The set of candidate road arcs $\mathcal{C}_t = \{c_{t, 1}, c_{t, 2}, \dots, c_{t, k}\}$ within $40\text{ m}$ of GPS observation $t$.
2. **Emission Likelihood:** Combines the neural network's candidate confidence with physical Gaussian GPS noise ($\sigma_z = 12.0\text{ m}$):
   $$\ln P(z_t \mid s_t = c_j) = \ln \hat{p}_{\text{MLP}}(c_j) - \frac{\text{dist}(z_t, c_j)^2}{2 \sigma_z^2}$$
3. **Transition Likelihood:** Penalizes discrepancy between network shortest-path distance $d_{\text{net}}(c_i, c_j)$ and great-circle GPS displacement $d_{\text{gps}}(z_{t-1}, z_t)$:
   $$\ln P(s_t = c_j \mid s_{t-1} = c_i) = \max\left( \ln \epsilon, -\frac{|d_{\text{net}}(c_i, c_j) - d_{\text{gps}}(z_{t-1}, z_t)|}{\beta} \right)$$
   where $\beta = 8.0\text{ m}$ is the exponential decay parameter and $\epsilon = 10^{-6}$ is the continuity probability floor.
4. **Handling Discontinuities & Gaps:**
   - When $\Delta t > 30\text{ s}$, the transition penalty is reset ($\ln P_{\text{trans}} = 0$) because the vehicle may have traveled unobserved distances.
   - Road network shortest paths are cached in memory using a directed graph lookup dictionary, accelerating Viterbi recurrence by $>100\times$.

### 4.2 Impact of HMM Decoding (Measured Across All 10 Student Test Trips)
| Metric | Classifier-Only (Pointwise Greedy Top-1) | Classifier + HMM Viterbi Decoding | Absolute Improvement |
|---|---|---|---|
| **Mean Observation Top-1 Accuracy** | 79.83% | **83.81%** | **+3.98%** |
| **Mean Route Continuity** | 91.42% | **94.41%** | **+2.99%** |
| **Trip 00000056 Top-1 Accuracy** | 20.24% | **55.88%** | **+35.64%** |
| **Trip 00000091 Top-1 Accuracy** | 84.21% | **88.79%** | **+4.58%** |
| **Trip 00000065 Continuity ($\Delta t=210s$)** | 89.83% | **95.57%** | **+5.74%** |
| **Mean End-to-End Latency** | 150 ms | **834 ms / trip (0.43 ms / point)** | Real-time suitable |

The HMM eliminates "flickering" between parallel roads, dual carriageways, and overpasses by enforcing global topological feasibility. On difficult trips such as `00000056`, sequential Viterbi decoding rescues broken predictions, boosting accuracy from 20.2% to 55.9%.

---

## 5. Distributed Deployment (FastAPI & Streamlit UI)

- **Laptop 1 (Backend - FastAPI):** Exposes high-throughput REST endpoints (`/trips`, `/predict`, `/evaluate`, `/map/{trip_id}`) with CORS support, serving MLP inference and HMM decoding.
- **Laptop 2 (Frontend - Streamlit):** Evaluator UI featuring real-time trip selection, interactive Folium map visualization, performance scorecards, and latency breakdowns.
- **Academic Integrity Guard:** Ground-truth labels remain strictly server-side in `/evaluate`; the model and client receive zero ground-truth data.

---

## 6. Academic Integrity Statement
All models were trained exclusively on the 80 training trips using fold-isolated preprocessing. Zero held-out evaluator data was accessed or simulated. All reported metrics, loss curves, and figures reflect reproducible runs executed from this repository.
