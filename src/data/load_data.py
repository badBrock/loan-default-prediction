"""Load raw nested-JSON and CSV data into flat DataFrames."""
import json
import pandas as pd

from src.utils import config


def load_nested_json(path) -> pd.DataFrame:
    """Flatten a nested JSON of shape ``list[list[dict | None]]`` into a DataFrame.

    Each applicant is a list of record dicts (with ``None`` padding). We drop the
    padding and concatenate all records; ``uid`` inside each record links back to
    the applicant.
    """
    with open(path) as f:
        raw = json.load(f)
    records = [rec for applicant in raw for rec in applicant if rec is not None]
    return pd.DataFrame(records)


def load_flag(path) -> pd.DataFrame:
    """Load the flag / label CSV (uid, NAME_CONTRACT_TYPE, [TARGET])."""
    return pd.read_csv(path)


def load_train():
    """Load the full training triple: (flag, accounts, enquiries)."""
    flag = load_flag(config.RAW_FLAG_TRAIN)
    acc = load_nested_json(config.RAW_ACCOUNTS_TRAIN)
    enq = load_nested_json(config.RAW_ENQUIRY_TRAIN)
    return flag, acc, enq


def load_test():
    """Load the full test triple: (flag, accounts, enquiries)."""
    flag = load_flag(config.RAW_FLAG_TEST)
    acc = load_nested_json(config.RAW_ACCOUNTS_TEST)
    enq = load_nested_json(config.RAW_ENQUIRY_TEST)
    return flag, acc, enq


if __name__ == "__main__":
    flag, acc, enq = load_train()
    print("flag:", flag.shape)
    print("acc :", acc.shape)
    print("enq :", enq.shape)