import math

from src.models.compare import build_table, to_markdown


def test_missing_metrics_render_as_na_and_numbers_pass_through():
    s = {
        "mlp": dict(
            family="Neural net",
            params=dict(hidden=64),
            cv_macro_f1_mean=0.9,
            cv_macro_f1_std=0.01,
            student_top1_acc=0.88,
            student_macro_f1=0.92,
        )
    }  # no log loss / continuity / timing
    t = build_table(s)
    assert math.isnan(float(t["Student Log Loss"].iloc[0]))
    md = to_markdown(t)
    assert "N/A" in md and "0.8800" in md and "0.9000 ± 0.0100" in md and "hidden=64" in md
