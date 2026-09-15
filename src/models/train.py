"""Train baseline and best (tuned) XGBoost models; persist to disk."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.utils import config
from src.features.build_features import get_feature_cols


def load_features():
    """Load engineered train features from disk."""
    return pd.read_csv(config.FEATURES_TRAIN)


def compute_scale_pos_weight(y) -> float:
    """Ratio of negatives to positives (for class-imbalance handling)."""
    return float((y == 0).sum() / (y == 1).sum())


def cv_auc(model, X, y, cv) -> float:
    """5-fold out-of-fold ROC-AUC."""
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return roc_auc_score(y, oof)


def train_baseline(train: pd.DataFrame, feature_cols):
    """Fit the baseline model and report CV AUC."""
    X, y = train[feature_cols], train[config.TARGET_COL]
    spw = compute_scale_pos_weight(y)

    params = {**config.BASELINE_XGB_PARAMS, "scale_pos_weight": spw}
    model = XGBClassifier(**params)

    cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    auc = cv_auc(model, X, y, cv)
    print(f"[baseline] 5-fold CV AUC: {auc:.5f}")

    model.fit(X, y)
    joblib.dump({"model": model, "features": feature_cols}, config.BASELINE_MODEL_PATH)
    print(f"[baseline] saved -> {config.BASELINE_MODEL_PATH}")
    return model, auc


def select_features(train: pd.DataFrame, feature_cols, params, cv):
    """Greedy importance-based feature pruning; keep the best-scoring subset."""
    X, y = train[feature_cols], train[config.TARGET_COL]

    # importance from a full-data fit
    imp_model = XGBClassifier(**params).fit(X, y)
    imp = pd.Series(imp_model.feature_importances_, index=feature_cols)

    best_cols = feature_cols
    best_auc = cv_auc(XGBClassifier(**params), X, y, cv)
    print(f"[fs] all {len(feature_cols)} feats: AUC={best_auc:.5f}")

    for q in [0.10, 0.20, 0.30]:
        keep = imp[imp > imp.quantile(q)].index.tolist()
        auc = cv_auc(XGBClassifier(**params), train[keep], y, cv)
        print(f"[fs] drop bottom {int(q*100)}% -> {len(keep)} feats: AUC={auc:.5f}")
        if auc > best_auc:
            best_auc, best_cols = auc, keep

    print(f"[fs] kept {len(best_cols)} feats  AUC={best_auc:.5f}")
    return best_cols, best_auc


def sweep_scale_pos_weight(train, feature_cols, params, cv):
    """Try a few scale_pos_weight values and keep the best."""
    X, y = train[feature_cols], train[config.TARGET_COL]
    spw = compute_scale_pos_weight(y)

    best_w, best_auc = spw, 0.0
    for w in [1.0, spw ** 0.5, spw * 0.5, spw]:
        auc = cv_auc(XGBClassifier(**{**params, "scale_pos_weight": w}), X, y, cv)
        print(f"[spw] spw={w:6.2f}  AUC={auc:.5f}")
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"[spw] best spw={best_w:.2f}  AUC={best_auc:.5f}")
    return best_w


def train_best(train: pd.DataFrame, feature_cols):
    """Full optimization pipeline: spw sweep -> feature selection -> final fit."""
    X, y = train[feature_cols], train[config.TARGET_COL]
    cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    params = dict(config.BEST_XGB_PARAMS)

    # 1. class weight
    best_w = sweep_scale_pos_weight(train, feature_cols, params, cv)
    params["scale_pos_weight"] = best_w

    # 2. feature selection
    best_cols, best_auc = select_features(train, feature_cols, params, cv)

    # 3. final fit on all data
    model = XGBClassifier(**params).fit(train[best_cols], y)
    joblib.dump(
        {"model": model, "features": best_cols, "cv_auc": best_auc, "params": params},
        config.BEST_MODEL_PATH,
    )
    print(f"[best] saved -> {config.BEST_MODEL_PATH}  (CV AUC={best_auc:.5f})")
    return model, best_cols, best_auc


def main():
    """Train both baseline and best models."""
    config.ensure_dirs()
    train = load_features()
    feature_cols = get_feature_cols(train)

    print("=" * 50)
    train_baseline(train, feature_cols)
    print("=" * 50)
    train_best(train, feature_cols)


if __name__ == "__main__":
    main()