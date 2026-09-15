"""Generate predictions on the test set and write a submission file."""
import joblib
import pandas as pd

from src.utils import config


def load_model(path=config.BEST_MODEL_PATH):
    """Load a serialized model bundle (model + feature list)."""
    return joblib.load(path)


def predict(bundle, test: pd.DataFrame) -> pd.Series:
    """Return P(default) for each applicant in `test`."""
    model, feats = bundle["model"], bundle["features"]

    # make sure every needed feature exists
    for c in feats:
        if c not in test.columns:
            test[c] = 0
    proba = model.predict_proba(test[feats])[:, 1]
    return pd.Series(proba, index=test.index, name=config.TARGET_COL)


def main(out_path=config.SUBMISSION_PATH):
    """Load best model, score test features, write submission CSV."""
    config.ensure_dirs()
    bundle = load_model()
    test = pd.read_csv(config.FEATURES_TEST)

    preds = predict(bundle, test)
    sub = pd.DataFrame({"uid": test["uid"], config.TARGET_COL: preds.values})
    sub.to_csv(out_path, index=False)

    print(f"Wrote {len(sub)} predictions -> {out_path}")
    print(sub.head())
    return sub


if __name__ == "__main__":
    main()