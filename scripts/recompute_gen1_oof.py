#!/usr/bin/env python3
"""Reconstruct leakage-controlled Gen-1 out-of-fold predictions.

The architecture, descriptors, outer folds, optimiser, loss, epoch limit and
batch size follow the best manual Gen-1 variant in PTM_1.0.ipynb.  Two small
but important validation corrections are made here: imputation/scaling are fit
inside each outer training fold, and early stopping uses an inner subset of
that training fold rather than the outer held-out fold.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "ml" / "gen1_development_source.csv"
OUT = ROOT / "outputs" / "recomputed_gen1"
FEATURES = ["mw", "r_exp", "tm", "tb"]
TARGET = "hl_cgf"
SEED = 42


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_model(n_features: int) -> Sequential:
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.30),
        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.20),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    return model


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = np.abs(pred - y)
    return {
        "n": int(len(y)),
        "mae_gpa": float(mean_absolute_error(y, pred)),
        "median_ae_gpa": float(np.median(err)),
        "rmse_gpa": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "pearson_r": float(np.corrcoef(y, pred)[0, 1]),
        "spearman_rho": float(spearmanr(y, pred).statistic),
        "fraction_abs_error_le_1_gpa": float(np.mean(err <= 1.0)),
        "max_abs_error_gpa": float(err.max()),
    }


def main() -> None:
    seed_everything(SEED)
    raw = pd.read_csv(INPUT)
    data = raw.loc[raw[TARGET].notna(), ["name", "smi", *FEATURES, TARGET]].reset_index(drop=True)
    x_raw = data[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[TARGET], errors="raise").to_numpy(float)

    pred = np.full(len(data), np.nan, dtype=float)
    fold_id = np.full(len(data), -1, dtype=int)
    epochs_used: list[int] = []
    outer = KFold(n_splits=10, shuffle=True, random_state=SEED)

    for fold, (outer_train, outer_test) in enumerate(outer.split(x_raw), start=1):
        # The held-out outer fold is never used for preprocessing or stopping.
        fit_idx, stop_idx = train_test_split(
            outer_train, test_size=0.15, random_state=SEED + fold
        )
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(imputer.fit_transform(x_raw[fit_idx]))
        x_stop = scaler.transform(imputer.transform(x_raw[stop_idx]))
        x_test = scaler.transform(imputer.transform(x_raw[outer_test]))

        seed_everything(SEED + fold)
        tf.keras.backend.clear_session()
        model = make_model(x_fit.shape[1])
        callback = EarlyStopping(
            monitor="val_loss", patience=40, min_delta=1e-4,
            restore_best_weights=True, verbose=0,
        )
        history = model.fit(
            x_fit, y[fit_idx], validation_data=(x_stop, y[stop_idx]),
            epochs=500, batch_size=16, callbacks=[callback], verbose=0,
        )
        pred[outer_test] = model.predict(x_test, verbose=0).ravel()
        fold_id[outer_test] = fold
        epochs_used.append(len(history.history["loss"]))
        print(f"fold {fold:02d}: n_test={len(outer_test)}, epochs={epochs_used[-1]}")

    if not np.isfinite(pred).all() or np.any(fold_id < 1):
        raise RuntimeError("OOF reconstruction did not generate exactly one prediction per row")

    output = pd.DataFrame({
        "compound": data["name"],
        "smiles": data["smi"],
        "historical_set": "set 1",
        "observed_hl_sim": y,
        "oof_prediction": pred,
        "oof_fold": fold_id,
        "absolute_error_gpa": np.abs(pred - y),
    })
    report = {
        "workflow": "reconstructed Gen-1 leakage-controlled OOF",
        "source_notebook": "historical PTM_1.0.ipynb, cells 23-24",
        "features": FEATURES,
        "target": "CGenFF sigma=0.2 HL_sim",
        "cv": {"type": "KFold", "n_splits": 10, "shuffle": True, "random_state": SEED},
        "validation_corrections": [
            "mean imputer and StandardScaler fitted within each outer training fold",
            "early stopping performed on an inner training subset, not the outer held-out fold",
        ],
        "architecture": [256, 128, 64, 32],
        "dropout": [0.30, 0.25, 0.20, 0.0],
        "l2_kernel": 0.001,
        "optimizer": "Adam default learning rate",
        "epochs_max": 500,
        "batch_size": 16,
        "epochs_used": epochs_used,
        "metrics": metrics(y, pred),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT / "gen1_reconstructed_oof_predictions.csv", index=False)
    (OUT / "gen1_reconstructed_oof_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
