"""Evaluation utilities: CV metrics, curves, and model comparison."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.utils import config
from src.features.build_features import get_feature_cols


def evaluate_cv(model, X, y, cv=None):
    """Return out-of-fold probabilities and ROC-AUC."""
    if cv is None:
        cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return oof, roc_auc_score(y, oof)


def plot_roc(y, oof, path=None):
    """Plot & optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y, oof)
    auc = roc_auc_score(y, oof)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120)
    plt.close()


def plot_pr(y, oof, path=None):
    """Plot & optionally save the Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y, oof)
    ap = average_precision_score(y, oof)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120)
    plt.close()


def classification_summary(y, oof, threshold=0.5):
    """Confusion matrix + text report at a fixed threshold."""
    preds = (oof >= threshold).astype(int)
    return {
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "report": classification_report(y, preds, digits=4),
    }


def compare_models(train: pd.DataFrame, feature_cols, spw):
    """CV-compare XGB / LGBM / CatBoost and a rank-blend ensemble."""
    X, y = train[feature_cols], train[config.TARGET_COL]
    cv = StratifiedKFold(config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    xgb = XGBClassifier(**{**config.BEST_XGB_PARAMS, "scale_pos_weight": spw})
    oof_x, auc_x = evaluate_cv(xgb, X, y, cv)

    lgb = LGBMClassifier(
        n_estimators=500, max_depth=3, num_leaves=8, learning_rate=0.05,
        subsample=0.75, colsample_bytree=0.65, reg_alpha=3.5, reg_lambda=8.5,
        min_child_samples=40, scale_pos_weight=spw,
        random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    oof_l, auc_l = evaluate_cv(lgb, X, y, cv)

    cat = CatBoostClassifier(
        iterations=500, depth=3, learning_rate=0.05, l2_leaf_reg=8,
        subsample=0.75, scale_pos_weight=spw, random_state=config.RANDOM_STATE,
        verbose=0, allow_writing_files=False,
    )
    oof_c = cross_val_predict(cat, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    auc_c = roc_auc_score(y, oof_c)

    rank_blend = (rankdata(oof_x) + rankdata(oof_l) + rankdata(oof_c)) / 3
    auc_blend = roc_auc_score(y, rank_blend)

    results = {
        "XGBoost": auc_x, "LightGBM": auc_l,
        "CatBoost": auc_c, "Rank-Blend": auc_blend,
    }
    for name, auc in results.items():
        print(f"{name:12s}: {auc:.5f}")
    return results


def main():
    """Evaluate the best model and write curves to reports/figures."""
    config.ensure_dirs()
    bundle = joblib.load(config.BEST_MODEL_PATH)
    train = pd.read_csv(config.FEATURES_TRAIN)
    feats = bundle["features"]
    X, y = train[feats], train[config.TARGET_COL]

    oof, auc = evaluate_cv(bundle["model"], X, y)
    print(f"Best model CV AUC: {auc:.5f}")

    plot_roc(y, oof, config.FIGURES_DIR / "roc_curve.png")
    plot_pr(y, oof, config.FIGURES_DIR / "pr_curve.png")

    summary = classification_summary(y, oof)
    print("Confusion matrix:", summary["confusion_matrix"])
    print(summary["report"])


if __name__ == "__main__":
    main()