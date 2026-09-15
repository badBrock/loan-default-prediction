"""Applicant-level feature engineering from accounts / enquiries / flag."""
import numpy as np
import pandas as pd

from src.utils import config
from src.data.load_data import load_train, load_test
from src.data.preprocessing import clean_accounts, clean_enquiries


# ----------------------------------------------------------------------
# ACCOUNT FEATURES
# ----------------------------------------------------------------------
def build_account_features(acc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned account rows into one feature row per applicant."""
    acc = clean_accounts(acc)
    g = acc.groupby("uid")

    num = g.agg(
        acc_n_accounts=("uid", "size"),
        acc_n_open=("is_open", "sum"),
        acc_n_zero_amt=("is_zero_amt", "sum"),
        acc_loan_amt_sum=("loan_amount", "sum"),
        acc_loan_amt_mean=("loan_amount", "mean"),
        acc_loan_amt_max=("loan_amount", "max"),
        acc_loan_amt_std=("loan_amount", "std"),
        acc_log_loan_sum=("log_loan_amount", "sum"),
        acc_log_loan_mean=("log_loan_amount", "mean"),
        acc_overdue_sum=("amount_overdue", "sum"),
        acc_overdue_max=("amount_overdue", "max"),
        acc_n_with_overdue=("has_overdue", "sum"),
        acc_n_credit_types=("credit_type", "nunique"),
        acc_days_since_open_min=("days_since_open", "min"),
        acc_days_since_open_max=("days_since_open", "max"),
        acc_days_since_open_mean=("days_since_open", "mean"),
        acc_days_since_close_min=("days_since_close", "min"),
    )

    # ratio features
    num["acc_open_ratio"] = num["acc_n_open"] / num["acc_n_accounts"]
    num["acc_overdue_ratio"] = num["acc_n_with_overdue"] / num["acc_n_accounts"]
    num["acc_overdue_amt_ratio"] = num["acc_overdue_sum"] / (num["acc_loan_amt_sum"] + 1)
    num["acc_zero_amt_ratio"] = num["acc_n_zero_amt"] / num["acc_n_accounts"]

    # per-credit-type counts + shares
    cat = (
        acc.groupby(["uid", "credit_grp"]).size()
        .unstack(fill_value=0)
        .add_prefix("acc_cnt_")
    )
    cat_share = cat.div(cat.sum(axis=1), axis=0).add_suffix("_share")

    return num.join(cat).join(cat_share).reset_index()


# ----------------------------------------------------------------------
# PAYMENT-HISTORY FEATURES
# ----------------------------------------------------------------------
def _parse_ph(s: str):
    """Parse a payment-history string into a list of monthly DPD ints.

    The string is groups of 3 digits per month, oldest → newest.
    """
    s = str(s)
    if s in ("", "nan") or len(s) < 3:
        return []
    n = len(s) - (len(s) % 3)
    return [int(s[i:i + 3]) for i in range(0, n, 3)]


def _ph_row_features(s: str) -> dict:
    """Per-account payment-history features."""
    vals = _parse_ph(s)
    if not vals:
        return dict(
            ph_n_months=0, ph_max_dpd=0, ph_mean_dpd=0, ph_sum_dpd=0,
            ph_n_late=0, ph_n_30=0, ph_n_90=0, ph_n_180=0, ph_n_360=0,
            ph_recent_dpd=0, ph_recent3_max=0, ph_late_ratio=0, ph_has_history=0,
        )
    arr = np.array(vals)
    return dict(
        ph_n_months=len(arr),
        ph_max_dpd=int(arr.max()),
        ph_mean_dpd=float(arr.mean()),
        ph_sum_dpd=int(arr.sum()),
        ph_n_late=int((arr > 0).sum()),
        ph_n_30=int((arr >= 30).sum()),
        ph_n_90=int((arr >= 90).sum()),
        ph_n_180=int((arr >= 180).sum()),
        ph_n_360=int((arr >= 360).sum()),
        ph_recent_dpd=int(arr[-1]),
        ph_recent3_max=int(arr[-3:].max()),
        ph_late_ratio=float((arr > 0).mean()),
        ph_has_history=1,
    )


def build_payment_features(acc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-account payment history into applicant-level features."""
    acc = acc.copy()
    ph = acc["payment_hist_string"].apply(_ph_row_features).apply(pd.Series)
    ph["uid"] = acc["uid"].values

    g = ph.groupby("uid")
    out = g.agg(
        pmt_max_dpd=("ph_max_dpd", "max"),
        pmt_mean_dpd=("ph_mean_dpd", "mean"),
        pmt_sum_dpd=("ph_sum_dpd", "sum"),
        pmt_n_late=("ph_n_late", "sum"),
        pmt_n_30=("ph_n_30", "sum"),
        pmt_n_90=("ph_n_90", "sum"),
        pmt_n_180=("ph_n_180", "sum"),
        pmt_n_360=("ph_n_360", "sum"),
        pmt_recent_dpd_max=("ph_recent_dpd", "max"),
        pmt_recent3_max=("ph_recent3_max", "max"),
        pmt_total_months=("ph_n_months", "sum"),
        pmt_late_ratio_max=("ph_late_ratio", "max"),
        pmt_late_ratio_mean=("ph_late_ratio", "mean"),
        pmt_n_acc_with_hist=("ph_has_history", "sum"),
    )
    out["pmt_ever_90plus"] = (out["pmt_n_90"] > 0).astype(int)
    out["pmt_ever_180plus"] = (out["pmt_n_180"] > 0).astype(int)
    out["pmt_ever_360plus"] = (out["pmt_n_360"] > 0).astype(int)
    return out.reset_index()


# ----------------------------------------------------------------------
# ENQUIRY FEATURES
# ----------------------------------------------------------------------
def build_enquiry_features(enq: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned enquiry rows into one feature row per applicant."""
    enq = clean_enquiries(enq)
    g = enq.groupby("uid")

    num = g.agg(
        enq_n=("uid", "size"),
        enq_amt_sum=("enquiry_amt", "sum"),
        enq_amt_mean=("enquiry_amt", "mean"),
        enq_amt_max=("enquiry_amt", "max"),
        enq_log_amt_sum=("log_enq_amt", "sum"),
        enq_log_amt_mean=("log_enq_amt", "mean"),
        enq_n_types=("enquiry_type", "nunique"),
        enq_days_since_last=("days_ago", "min"),
        enq_days_since_first=("days_ago", "max"),
        enq_n_30d=("in_30d", "sum"),
        enq_n_90d=("in_90d", "sum"),
        enq_n_180d=("in_180d", "sum"),
        enq_n_365d=("in_365d", "sum"),
    )
    num["enq_span_days"] = num["enq_days_since_first"] - num["enq_days_since_last"]

    cat = (
        enq.groupby(["uid", "enq_grp"]).size()
        .unstack(fill_value=0)
        .add_prefix("enq_cnt_")
    )
    return num.join(cat).reset_index()


# ----------------------------------------------------------------------
# MASTER TABLE
# ----------------------------------------------------------------------
def build_master(flag: pd.DataFrame, acc: pd.DataFrame, enq: pd.DataFrame) -> pd.DataFrame:
    """Merge all feature blocks onto the flag table (one row per applicant)."""
    m = flag.copy()
    m["is_cash_loan"] = (m["NAME_CONTRACT_TYPE"] == "Cash loans").astype(int)

    m = m.merge(build_account_features(acc), on="uid", how="left")
    m = m.merge(build_payment_features(clean_accounts(acc)), on="uid", how="left")
    m = m.merge(build_enquiry_features(enq), on="uid", how="left")

    # missingness flag before filling
    m["has_accounts"] = m["acc_n_accounts"].notna().astype(int)

    acc_cols = [c for c in m.columns if c.startswith(("acc_", "pmt_"))]
    m[acc_cols] = m[acc_cols].fillna(0)
    enq_cols = [c for c in m.columns if c.startswith("enq_")]
    m[enq_cols] = m[enq_cols].fillna(0)
    m = m.fillna(0)
    return m


def get_feature_cols(train: pd.DataFrame):
    """Return the list of usable feature columns (excludes id/target/meta)."""
    return [c for c in train.columns if c not in config.META_COLS]


def align_test(train: pd.DataFrame, test: pd.DataFrame, feature_cols):
    """Ensure test has exactly the train feature columns (fill missing with 0)."""
    for c in feature_cols:
        if c not in test.columns:
            test[c] = 0
    return test[["uid"] + feature_cols]


def main():
    """Build and persist engineered train/test feature CSVs."""
    config.ensure_dirs()

    flag_tr, acc_tr, enq_tr = load_train()
    train = build_master(flag_tr, acc_tr, enq_tr)

    flag_te, acc_te, enq_te = load_test()
    test = build_master(flag_te, acc_te, enq_te)

    feature_cols = get_feature_cols(train)
    test = align_test(train, test, feature_cols)

    train.to_csv(config.FEATURES_TRAIN, index=False)
    test.to_csv(config.FEATURES_TEST, index=False)

    print(f"Train: {train.shape} -> {config.FEATURES_TRAIN}")
    print(f"Test : {test.shape} -> {config.FEATURES_TEST}")
    print(f"Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()