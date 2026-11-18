| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy (student-test) | Route Continuity | Student-Test Macro F1 | Log Loss (student-test) | Inference Time (ms/point) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | C=1.0, max_iter=1000, solver=lbfgs | 0.9325 ± 0.0078 | 0.8533 | 0.9265 | 0.9138 | 0.1428 | 0.0010 |
| Decision Tree | Decision tree (single, non-ensemble) | max_depth=10, min_samples_leaf=20 | 0.9378 ± 0.0076 | 0.8757 | 0.9371 | 0.9269 | 0.1071 | 0.0002 |
| Random Forest | Bagging ensemble (trees) | n_estimators=100, max_depth=None, min_samples_leaf=20, max_features=sqrt, max_samples=0.25 | 0.9404 ± 0.0074 | 0.8740 | 0.9389 | 0.9259 | 0.0993 | 0.0295 |
| MLP | Neural network (MLP) | hidden_layer_sizes=[64, 32], activation=relu, solver=adam, alpha=0.0001, learning_rate_init=0.001, batch_size=1024, epochs=20 | 0.9374 ± 0.0074 | 0.8809 | 0.9431 | 0.9300 | 0.1052 | 0.0021 |
