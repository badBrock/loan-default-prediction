"""Sanity-check / validation helpers for the raw and engineered data."""
import pandas as pd


def validate_accounts(acc: pd.DataFrame) -> dict:
    """Return a dict of validation metrics for the accounts table."""
    acc = acc.copy()
    acc["open_date"] = pd.to_datetime(acc["open_date"], errors="coerce")
    acc["closed_date"] = pd.to_datetime(acc["closed_date"], errors="coerce")

    bad_closed = (acc["closed_date"] < acc["open_date"]).sum()
    return {
        "n_rows": len(acc),
        "n_uids": acc["uid"].nunique(),
        "missing_closed_date_pct": acc["closed_date"].isna().mean(),
        "invalid_closed_lt_open": int(bad_closed),
        "zero_loan_amount": int((pd.to_numeric(acc["loan_amount"], errors="coerce") == 0).sum()),
        "open_date_max": acc["open_date"].max(),
        "closed_date_max": acc["closed_date"].max(),
    }


def validate_enquiries(enq: pd.DataFrame) -> dict:
    """Return a dict of validation metrics for the enquiries table."""
    enq = enq.copy()
    enq["enquiry_date"] = pd.to_datetime(enq["enquiry_date"], errors="coerce")
    return {
        "n_rows": len(enq),
        "n_uids": enq["uid"].nunique(),
        "missing_pct": enq.isna().mean().to_dict(),
        "date_min": enq["enquiry_date"].min(),
        "date_max": enq["enquiry_date"].max(),
        "zero_amount": int((pd.to_numeric(enq["enquiry_amt"], errors="coerce") == 0).sum()),
    }


def validate_payment_history(acc: pd.DataFrame) -> dict:
    """Check the payment-history-string encoding assumptions."""
    ph = acc["payment_hist_string"].astype(str)
    lengths = ph.str.len()
    return {
        "pct_len_divisible_by_3": float((lengths % 3 == 0).mean()),
        "n_empty": int((ph == "").sum()),
        "n_nan_str": int((ph == "nan").sum()),
        "max_months": int((lengths // 3).max()),
    }


def check_uid_overlap(flag: pd.DataFrame, acc: pd.DataFrame, enq: pd.DataFrame) -> dict:
    """Fraction of flag applicants that have accounts / enquiries."""
    return {
        "has_accounts": flag["uid"].isin(acc["uid"]).mean(),
        "has_enquiries": flag["uid"].isin(enq["uid"]).mean(),
        "acc_uids_in_flag": acc["uid"].isin(flag["uid"]).mean(),
        "enq_uids_in_flag": enq["uid"].isin(flag["uid"]).mean(),
    }


def assert_feature_frame(df: pd.DataFrame, feature_cols, target_col="TARGET"):
    """Guardrail assertions before training."""
    assert df["uid"].is_unique, "uid must be unique per applicant"
    assert target_col not in feature_cols, "target leaked into features!"
    assert not df[feature_cols].isna().any().any(), "NaNs present in features"