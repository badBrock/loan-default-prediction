"""Clean raw tables: dtypes, invalid dates, derived per-row fields."""
import numpy as np
import pandas as pd

from src.utils import config


def clean_accounts(acc: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, drop invalid closed_dates, and add per-row helper columns."""
    acc = acc.copy()

    acc["open_date"] = pd.to_datetime(acc["open_date"], errors="coerce")
    acc["closed_date"] = pd.to_datetime(acc["closed_date"], errors="coerce")
    acc["loan_amount"] = pd.to_numeric(acc["loan_amount"], errors="coerce")
    acc["amount_overdue"] = pd.to_numeric(acc["amount_overdue"], errors="coerce")

    # invalid: closed before opened
    bad = acc["closed_date"] < acc["open_date"]
    acc.loc[bad, "closed_date"] = pd.NaT

    acc["is_open"] = acc["closed_date"].isna().astype(int)
    acc["is_zero_amt"] = (acc["loan_amount"] == 0).astype(int)
    acc["has_overdue"] = (acc["amount_overdue"] > 0).astype(int)
    acc["days_since_open"] = (config.REF_DATE - acc["open_date"]).dt.days
    acc["days_since_close"] = (config.REF_DATE - acc["closed_date"]).dt.days
    acc["log_loan_amount"] = np.log1p(acc["loan_amount"].clip(lower=0))

    acc["credit_grp"] = np.where(
        acc["credit_type"].isin(config.KEEP_CREDIT_TYPES),
        acc["credit_type"], "Other",
    )
    return acc


def clean_enquiries(enq: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and add per-row recency / log-amount fields."""
    enq = enq.copy()
    enq["enquiry_date"] = pd.to_datetime(enq["enquiry_date"], errors="coerce")
    enq["enquiry_amt"] = pd.to_numeric(enq["enquiry_amt"], errors="coerce")
    enq["log_enq_amt"] = np.log1p(enq["enquiry_amt"].clip(lower=0))
    enq["days_ago"] = (config.REF_DATE - enq["enquiry_date"]).dt.days

    for w in [30, 90, 180, 365]:
        enq[f"in_{w}d"] = (enq["days_ago"] <= w).astype(int)

    enq["enq_grp"] = np.where(
        enq["enquiry_type"].isin(config.KEEP_ENQ_TYPES),
        enq["enquiry_type"], "Other",
    )
    return enq