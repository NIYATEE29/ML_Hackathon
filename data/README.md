# Data

Raw data is not committed (size). Place the provided `Student folder/` contents here:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

- `train` (80 trips): fitting, cross-validation, optimisation.
- `student_test` (10 trips): development-stage comparison.
- Evaluator-test data must never be placed in or used from this repo during development.
