#!/usr/bin/env python3
"""Recompute the final Gen-2 deployable model's historical 10x10 OOF values.

Unlike the earlier general reconstruction, this mirrors PTM_gen2_2.4.ipynb:
the target QuantileTransformer is rebuilt inside every fold and its number of
quantiles is derived from that fold's training-set size. Existing historical
files are not overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "ml" / "gen2_development_157.csv"
OUT = ROOT / "outputs" / "recomputed_gen2"
MODEL_OUT = OUT
SEED = 42
N_SPLITS = 10
N_REPEATS = 10
FEATURES = [
    "mw", "rho_exp", "tm", "tb", "ri", "pol", "i_vol", "alpha",
    "nat", "non_h", "n_N", "n_P", "n_O", "n_S", "n_F", "n_Cl",
    "n_Br", "n_I", "vp", "kappa", "tc", "pc", "omega", "triple", "tg",
]


def make_model(n_total_samples: int) -> TransformedTargetRegressor:
    min_train = int((N_SPLITS - 1) * n_total_samples // N_SPLITS)
    n_quantiles = max(16, min(1000, max(2, min_train - 1)))
    regressor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=800, max_features=.6, min_samples_leaf=1,
            min_samples_split=3, bootstrap=False, random_state=SEED,
            n_jobs=-1,
        )),
    ])
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=QuantileTransformer(
            output_distribution="normal", n_quantiles=n_quantiles,
            subsample=1_000_000, random_state=SEED,
        ),
    )


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "pcc": float(np.corrcoef(y, p)[0, 1]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    frame.rename(columns={"smiles": "smile", "hl_sim_forcefield_mean_gpa": "sigma_def1_avg"}, inplace=True)
    work = frame.loc[frame.sigma_def1_avg.notna()].reset_index(drop=True)
    x = work[FEATURES]
    y = work.sigma_def1_avg.to_numpy(float)

    oof = np.full((len(work), N_REPEATS), np.nan)
    seen = np.zeros(len(work), dtype=int)
    fold_metrics = []
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for fold, (train, test) in enumerate(cv.split(x, y), 1):
        # This fold-local construction is the detail used in the old notebook.
        model = make_model(len(train))
        model.fit(x.iloc[train], y[train])
        pred = model.predict(x.iloc[test])
        for idx, value in zip(test, pred):
            oof[idx, seen[idx]] = value
            seen[idx] += 1
        fold_metrics.append(scores(y[test], pred))
        if fold == 1 or fold % 10 == 0:
            print(f"completed {fold}/{N_SPLITS * N_REPEATS} folds", flush=True)
    if not np.all(seen == N_REPEATS):
        raise RuntimeError(f"Unexpected OOF counts: {np.unique(seen, return_counts=True)}")

    oof_mean = oof.mean(axis=1)
    oof_sd = oof.std(axis=1, ddof=1)
    final = make_model(len(work)).fit(x, y)
    train_prediction = final.predict(x)

    table = pd.DataFrame({
        "compound": work.name, "smiles": work.smile,
        "historical_set": work.historical_set,
        "observed_hl_sim": y,
        "train_prediction": train_prediction,
        "oof_prediction": oof_mean,
        "oof_prediction_sd": oof_sd,
    })
    for repeat in range(N_REPEATS):
        table[f"oof_repeat_{repeat + 1:02d}"] = oof[:, repeat]
    table.to_csv(OUT / "quick_et_qttr_exact_oof_predictions.csv", index=False)

    fold_mae = np.array([m["mae"] for m in fold_metrics])
    fold_pcc = np.array([m["pcc"] for m in fold_metrics])
    report = {
        "workflow": "historical fold-local ET+QTTR construction",
        "features": FEATURES,
        "target": "row-wise mean available CGenFF/OPLS-AA sigma=0.1 HL_sim",
        "cv": {"type": "RepeatedKFold", "n_splits": N_SPLITS,
               "n_repeats": N_REPEATS, "random_state": SEED},
        "train": scores(y, train_prediction),
        "aggregated_oof": scores(y, oof_mean),
        "fold_summary": {
            "mae_mean": float(fold_mae.mean()),
            "mae_ci95_half_width": float(1.96 * fold_mae.std(ddof=1) / np.sqrt(len(fold_mae))),
            "pcc_mean": float(fold_pcc.mean()),
            "pcc_ci95_half_width": float(1.96 * fold_pcc.std(ddof=1) / np.sqrt(len(fold_pcc))),
        },
        "quantiles": {
            "per_oof_fold": int(make_model(len(work) * 9 // 10).transformer.n_quantiles),
            "final_fit": int(make_model(len(work)).transformer.n_quantiles),
        },
    }
    (OUT / "quick_et_qttr_exact_oof_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump({"model": final, "features": FEATURES, "report": report},
                MODEL_OUT / "quick_et_qttr_exact_final_model.joblib")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
