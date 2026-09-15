"""Unit tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    _parse_ph, _ph_row_features,
    build_account_features, build_payment_features,
    build_enquiry_features, build_master, get_feature_cols, align_test,
)


# ----------------------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------------------
@pytest.fixture
def accounts():
    return pd.DataFrame({
        "credit_type": ["Consumer credit", "Consumer credit", "Microloan"],
        "loan_amount": [1000.0, 2000.0, 500.0],
        "amount_overdue": [0.0, 100.0, 0.0],
        "open_date": ["2020-01-01", "2019-01-01", "2018-06-01"],
        "closed_date": ["2020-06-01", None, "2019-06-01"],
        "payment_hist_string": ["000000", "030090000", "000"],
        "uid": ["A1", "A1", "A2"],
    })


@pytest.fixture
def enquiries():
    return pd.DataFrame({
        "enquiry_type": ["Cash loans", "Revolving loans", "Microloan"],
        "enquiry_amt": [1000, 2000, 500],
        "enquiry_date": ["2020-12-01", "2020-06-01", "2019-01-01"],
        "uid": ["A1", "A1", "A2"],
    })


@pytest.fixture
def flag():
    return pd.DataFrame({
        "uid": ["A1", "A2", "A3"],           # A3 has no accounts / enquiries
        "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
        "TARGET": [0, 1, 0],
    })


# ----------------------------------------------------------------------
# PAYMENT-HISTORY PARSING
# ----------------------------------------------------------------------
def test_parse_ph_basic():
    assert _parse_ph("000030090") == [0, 30, 90]


def test_parse_ph_empty():
    assert _parse_ph("") == []
    assert _parse_ph("nan") == []
    assert _parse_ph("00") == []            # < 3 chars


def test_parse_ph_truncates_incomplete_chunk():
    # 7 chars -> use first 6 (2 months)
    assert _parse_ph("0300901") == [30, 90]


def test_ph_row_features_no_history():
    f = _ph_row_features("")
    assert f["ph_has_history"] == 0
    assert f["ph_max_dpd"] == 0


def test_ph_row_features_values():
    f = _ph_row_features("000030090")
    assert f["ph_n_months"] == 3
    assert f["ph_max_dpd"] == 90
    assert f["ph_n_late"] == 2
    assert f["ph_n_90"] == 1
    assert f["ph_recent_dpd"] == 90         # last month
    assert f["ph_has_history"] == 1


# ----------------------------------------------------------------------
# ACCOUNT FEATURES
# ----------------------------------------------------------------------
def test_build_account_features_counts(accounts):
    feats = build_account_features(accounts)
    a1 = feats.loc[feats["uid"] == "A1"].iloc[0]
    assert a1["acc_n_accounts"] == 2
    assert a1["acc_n_open"] == 1            # one row has closed_date None
    assert a1["acc_n_with_overdue"] == 1


def test_build_account_features_ratios(accounts):
    feats = build_account_features(accounts)
    a1 = feats.loc[feats["uid"] == "A1"].iloc[0]
    assert a1["acc_open_ratio"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# PAYMENT FEATURES
# ----------------------------------------------------------------------
def test_build_payment_features(accounts):
    from src.data.preprocessing import clean_accounts
    feats = build_payment_features(clean_accounts(accounts))
    a1 = feats.loc[feats["uid"] == "A1"].iloc[0]
    assert a1["pmt_max_dpd"] == 90
    assert a1["pmt_ever_90plus"] == 1


# ----------------------------------------------------------------------
# ENQUIRY FEATURES
# ----------------------------------------------------------------------
def test_build_enquiry_features(enquiries):
    feats = build_enquiry_features(enquiries)
    a1 = feats.loc[feats["uid"] == "A1"].iloc[0]
    assert a1["enq_n"] == 2
    assert a1["enq_n_types"] == 2


# ----------------------------------------------------------------------
# MASTER TABLE
# ----------------------------------------------------------------------
def test_build_master_no_nans(flag, accounts, enquiries):
    m = build_master(flag, accounts, enquiries)
    assert not m.isna().any().any()


def test_build_master_missing_flag(flag, accounts, enquiries):
    m = build_master(flag, accounts, enquiries)
    a3 = m.loc[m["uid"] == "A3"].iloc[0]
    assert a3["has_accounts"] == 0          # A3 has no accounts
    assert a3["acc_n_accounts"] == 0        # filled with 0


def test_build_master_is_cash_loan(flag, accounts, enquiries):
    m = build_master(flag, accounts, enquiries)
    assert m.loc[m["uid"] == "A1", "is_cash_loan"].iloc[0] == 1
    assert m.loc[m["uid"] == "A2", "is_cash_loan"].iloc[0] == 0


def test_get_feature_cols_excludes_meta(flag, accounts, enquiries):
    m = build_master(flag, accounts, enquiries)
    cols = get_feature_cols(m)
    assert "TARGET" not in cols
    assert "uid" not in cols
    assert "NAME_CONTRACT_TYPE" not in cols


def test_align_test_adds_missing_cols(flag, accounts, enquiries):
    train = build_master(flag, accounts, enquiries)
    cols = get_feature_cols(train)
    # simulate a test frame missing one column
    test = train.drop(columns=[cols[0]]).copy()
    aligned = align_test(train, test, cols)
    assert list(aligned.columns) == ["uid"] + cols
    assert (aligned[cols[0]] == 0).all()