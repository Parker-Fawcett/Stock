"""Tree baseline with walk-forward + purge/embargo + calibration.

Why trees first: his XGBoost beat a year of LSTM work on the same data
(70Dj8q_ntCg) and literature agrees trees win on tabular panels.
LSTM is the wrong tool until sequences actually help.

Purge/embargo (Lopez de Prado; his V5 1F0gYkk7YYw):
 rows whose label window [Date, label_end] reaches into the next split
 are dropped -- both train-vs-val and val-vs-test, since validation
 labels feed the calibrator/early-stopping just like training labels
 feed the fit. embargo_days pads the cut by an extra buffer beyond the
 deterministic label-horizon overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import HORIZON, model_cols


@dataclass
class Fold:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def walk_forward(
    panel: pd.DataFrame,
    train_years: float = 3.0,
    val_months: int = 6,
    test_months: int = 6,
    embargo_days: int = HORIZON,
) -> list[Fold]:
    dates = np.sort(panel["Date"].unique())
    start = dates[0] + pd.to_timedelta(int(train_years * 365), unit="D")
    folds: list[Fold] = []
    step = pd.to_timedelta(test_months * 30, unit="D")
    val_len = pd.to_timedelta(val_months * 30, unit="D")
    test_len = pd.to_timedelta(test_months * 30, unit="D")
    train_len = pd.to_timedelta(int(train_years * 365), unit="D")
    embargo = pd.to_timedelta(embargo_days, unit="D")
    t = start
    while t + val_len + test_len <= dates[-1]:
        te0, te1 = t + val_len, t + val_len + test_len
        va0, va1 = t, t + val_len
        tr0, tr1 = t - train_len, t
        test = panel[(panel["Date"] >= te0) & (panel["Date"] < te1)]
        val = panel[(panel["Date"] >= va0) & (panel["Date"] < va1)]
        train = panel[(panel["Date"] >= tr0) & (panel["Date"] < va0)]
        # PURGE: drop rows whose actual (trading-day) label window reaches
        # into or past the next split's start, plus an embargo buffer.
        train = train[train["label_end"] < va0 - embargo]
        # Validation labels feed the calibrator/early-stopping, so they must
        # not see outcomes inside the test window either.
        val = val[val["label_end"] < te0 - embargo]
        if len(train) > 500 and len(val) > 100 and len(test) > 100:
            folds.append(Fold(train, val, test))
        t += step
    return folds


def fit_fold(train: pd.DataFrame, val: pd.DataFrame, seed: int = 0):
    cols = model_cols(train)
    Xtr, ytr = train[cols].values, train["label"].values
    Xva = val[cols].values
    use_lgbm = False
    try:
        import lightgbm as lgb  # type: ignore

        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xva, label=val["label"].values, reference=dtrain)
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "seed": seed, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        }
        booster = lgb.train(params, dtrain, num_boost_round=500,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(50, verbose=False)])

        class _LGBMWrapper:
            def predict_proba(self, X):
                p = np.clip(booster.predict(X), 1e-6, 1 - 1e-6)
                return np.vstack([1 - p, p]).T

        # Booster is not an sklearn estimator, so skip isotonic here;
        # probabilities are used raw (note in evaluate).
        return _LGBMWrapper()
    except Exception:  # lightgbm missing -> sklearn, still a tree baseline
        use_lgbm = False
    _ = use_lgbm
    base = HistGradientBoostingClassifier(
        max_iter=300, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, random_state=seed,
    )
    base.fit(Xtr, ytr)
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except Exception:  # noqa: BLE001 - older sklearn
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(Xva, val["label"].values)
    return cal
