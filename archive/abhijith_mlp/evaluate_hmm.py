import time
from pathlib import Path
import pandas as pd
import numpy as np

from archive.legacy_config import TABLES_DIR, MODELS_DIR
from src.data.loader import list_trips
from archive.abhijith_mlp.pipeline import PathSnapPipeline

def run_hmm_comparison() -> pd.DataFrame:
    """
    Evaluates Classifier-Only vs. Classifier + HMM Viterbi decoding
    across all 10 student-testing trips.
    Saves results to results/tables/classifier_vs_hmm.csv.
    """
    model_path = MODELS_DIR / "mlp_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the MLP first.")

    pipeline = PathSnapPipeline(model_path=model_path)
    test_trips = list_trips("student_test")
    print(f"--- Running Classifier vs. HMM Evaluation on {len(test_trips)} Student Test Trips ---")

    records = []
    for tid in test_trips:
        t0 = time.time()
        res = pipeline.run_trip(trip_id=tid, split="student_test", use_cached_features=True)
        elapsed = time.time() - t0

        records.append({
            "trip_id": tid,
            "num_points": res["num_points"],
            "classifier_top1": res.get("classifier_top1_accuracy", 0.0),
            "hmm_top1": res.get("hmm_top1_accuracy", 0.0),
            "classifier_continuity": res["classifier_continuity"],
            "hmm_continuity": res["hmm_continuity"],
            "route_jaccard": res.get("route_jaccard_similarity", 0.0),
            "latency_ms": res["timings"]["total_pipeline_ms"]
        })

        print(f"Trip {tid}: Clf Top-1 = {res.get('classifier_top1_accuracy', 0)*100:.2f}%, "
              f"HMM Top-1 = {res.get('hmm_top1_accuracy', 0)*100:.2f}%, "
              f"Clf Cont = {res['classifier_continuity']*100:.1f}%, "
              f"HMM Cont = {res['hmm_continuity']*100:.1f}% ({elapsed:.2f}s)")

    df = pd.DataFrame(records)

    # Compute Summary Row
    summary = pd.DataFrame([{
        "trip_id": "MEAN",
        "num_points": int(df["num_points"].mean()),
        "classifier_top1": round(df["classifier_top1"].mean(), 4),
        "hmm_top1": round(df["hmm_top1"].mean(), 4),
        "classifier_continuity": round(df["classifier_continuity"].mean(), 4),
        "hmm_continuity": round(df["hmm_continuity"].mean(), 4),
        "route_jaccard": round(df["route_jaccard"].mean(), 4),
        "latency_ms": round(df["latency_ms"].mean(), 2)
    }])

    df_full = pd.concat([df, summary], ignore_index=True)
    csv_path = TABLES_DIR / "classifier_vs_hmm.csv"
    df_full.to_csv(csv_path, index=False)
    print(f"\n==========================================")
    print(f"Classifier vs. HMM Overall Results:")
    print(f"  Mean Classifier Top-1 Accuracy: {summary['classifier_top1'].values[0]*100:.2f}%")
    print(f"  Mean HMM Top-1 Accuracy:        {summary['hmm_top1'].values[0]*100:.2f}%")
    print(f"  Mean Classifier Continuity:     {summary['classifier_continuity'].values[0]*100:.1f}%")
    print(f"  Mean HMM Continuity:            {summary['hmm_continuity'].values[0]*100:.1f}%")
    print(f"Saved comparison table to {csv_path}")
    print(f"==========================================")
    return df_full

if __name__ == "__main__":
    run_hmm_comparison()
