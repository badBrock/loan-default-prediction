"""Unit tests for data loading, cleaning, and validation."""
import json
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_nested_json
from src.data.preprocessing import clean_accounts, clean_enquiries
from src.data.validation import validate_payment_history, check_uid_overlap


# ----------------------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------------------
@pytest.fixture
def nested_json_file(tmp_path):
    data = [
        [{"credit_type": "Consumer credit", "loan_amount": 1000.0,
          "amount_overdue": 0.0, "open_date": "2020-01-01",
          "closed_date": "2020-06-01", "payment_hist_string": "000000",
          "uid": "A1"}, None, None],
        [{"credit_type": "Credit card", "loan_amount": 500.0,
          "amount_overdue": 10.0, "open_date": "2019-01-01",
          "closed_date": None, "payment_hist_string": "030000",
          "uid": "A2"}],
    ]
    p = tmp_path / "nested.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def raw_accounts():
    return pd.DataFrame({
        "credit_type": ["Consumer credit", "Microloan", "Weird type"],
        "loan_amount": [1000.0, 0.0, 2000.0],
        "amount_overdue": [0.0, 50.0, 0.0],
        "open_date": ["2020-01-01", "2019-05-01", "2018-01-01"],
        # 2nd row has closed < open -> should become NaT
        "closed_date": [None, "2019-01-01", "2019-01-01"],
        "payment_hist_string": ["000000", "030060", ""],
        "uid": ["A1", "A2", "A3"],
    })


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------
def test_load_nested_json_flattens_and_drops_none(nested_json_file):
    df = load_nested_json(nested_json_file)
    assert len(df) == 2                       # 2 real records, None dropped
    assert set(df["uid"]) == {"A1", "A2"}
    assert "payment_hist_string" in df.columns


def test_clean_accounts_dtypes(raw_accounts):
    acc = clean_accounts(raw_accounts)
    assert pd.api.types.is_datetime64_any_dtype(acc["open_date"])
    assert pd.api.types.is_numeric_dtype(acc["loan_amount"])


def test_clean_accounts_invalid_closed_date(raw_accounts):
    acc = clean_accounts(raw_accounts)
    # A2: closed 2019-01-01 < open 2019-05-01 -> NaT
    a2 = acc.loc[acc["uid"] == "A2", "closed_date"].iloc[0]
    assert pd.isna(a2)


def test_clean_accounts_flags(raw_accounts):
    acc = clean_accounts(raw_accounts)
    assert acc.loc[acc["uid"] == "A2", "is_zero_amt"].iloc[0] == 1
    assert acc.loc[acc["uid"] == "A2", "has_overdue"].iloc[0] == 1
    assert acc.loc[acc["uid"] == "A1", "is_open"].iloc[0] == 1  # closed_date None


def test_clean_accounts_credit_grouping(raw_accounts):
    acc = clean_accounts(raw_accounts)
    assert acc.loc[acc["uid"] == "A3", "credit_grp"].iloc[0] == "Other"
    assert acc.loc[acc["uid"] == "A1", "credit_grp"].iloc[0] == "Consumer credit"


def test_clean_enquiries_recency_flags():
    enq = pd.DataFrame({
        "enquiry_type": ["Cash loans", "Microloan"],
        "enquiry_amt": [1000, 2000],
        "enquiry_date": ["2020-12-15", "2018-01-01"],
        "uid": ["A1", "A2"],
    })
    out = clean_enquiries(enq)
    assert "in_30d" in out.columns
    assert out.loc[0, "in_30d"] == 1     # 2020-12-15 within 30d of 2021-01-01
    assert out.loc[1, "in_365d"] == 0    # 2018 far in the past


def test_validate_payment_history():
    acc = pd.DataFrame({"payment_hist_string": ["000000", "030", ""]})
    res = validate_payment_history(acc)
    assert res["pct_len_divisible_by_3"] == 1.0
    assert res["n_empty"] == 1


def test_check_uid_overlap():
    flag = pd.DataFrame({"uid": ["A1", "A2", "A3"]})
    acc = pd.DataFrame({"uid": ["A1", "A2"]})
    enq = pd.DataFrame({"uid": ["A1", "A2", "A3"]})
    res = check_uid_overlap(flag, acc, enq)
    assert res["has_accounts"] == pytest.approx(2 / 3)
    assert res["has_enquiries"] == 1.0