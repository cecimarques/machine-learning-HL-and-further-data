#!/usr/bin/env python3
"""Validate the standalone PTM-paper-3 release without the source repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]


def metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    return {
        "r2": r2_score(y, p),
        "rmse_gpa": np.sqrt(mean_squared_error(y, p)),
        "mae_gpa": mean_absolute_error(y, p),
        "spearman_rho": spearmanr(y, p).statistic,
    }


def close(actual: dict[str, float], expected: dict[str, float], atol: float = 1e-12) -> None:
    for key in actual:
        if not np.isclose(actual[key], expected[key], atol=atol, rtol=0):
            raise AssertionError(f"{key}: {actual[key]} != {expected[key]}")


def main() -> None:
    report = json.loads((ROOT / "validation/validation_report.json").read_text())
    for name, expected in report["reference_pdf_sha256"].items():
        actual = hashlib.sha256((ROOT / "reference_figures" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Reference PDF hash mismatch: {name}")

    a = pd.read_csv(ROOT / "data/figures/main/Fig3/panel_a_oof_predictions.csv")
    b = pd.read_csv(ROOT / "data/figures/main/Fig3/panel_b_external_screen_ecdf.csv")
    c = pd.read_csv(ROOT / "data/figures/main/Fig3/panel_c_prospective_validation.csv")
    assert (len(a), len(b), len(c), int(b.selected_for_set3.sum())) == (156, 264, 17, 17)
    close(metrics(a.observed_hl_sim, a.oof_prediction), report["figure_3_metrics"]["panel_a_h2o2_excluded"])
    close(metrics(c.observed_sigma1_mean, c.sigma_def1_pred_et_qttr), report["figure_3_metrics"]["panel_c_set3"])

    dev = pd.read_csv(ROOT / "data/ml/gen2_development_157.csv")
    oof = pd.read_csv(ROOT / "data/ml/gen2_repeated_oof_predictions.csv")
    assert len(dev) == len(oof) == 157
    assert dev.name.tolist() == oof.compound.tolist()
    assert np.allclose(dev.hl_sim_forcefield_mean_gpa, oof.observed_hl_sim)

    index = pd.read_csv(ROOT / "data/FIGURE_DATA_INDEX.csv")
    assert len(index) == 26
    for row in index.itertuples(index=False):
        assert len(pd.read_csv(ROOT / row.path)) == row.rows

    print("PASS: PDF hashes, 26 panel/series datasets, Fig. 3 metrics/counts, and exact Gen-2 row alignment.")


if __name__ == "__main__":
    main()
