import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import FEATURES, load_raw_data, prepare_features, clean_training_data


def test_valid_input_has_all_features():
    X, warnings = prepare_features(pd.DataFrame([{"tenure": 5, "MonthlyCharges": 50.0, "Contract": "One year"}]))
    assert list(X.columns) == FEATURES
    assert len(warnings) >= 1  # optional columns were missing


def test_blank_total_charges_is_estimated():
    df = pd.DataFrame([{"tenure": 10, "MonthlyCharges": 20.0, "Contract": "Two year", "TotalCharges": " "}])
    X, _ = prepare_features(df)
    assert X["TotalCharges"].iloc[0] == 200.0


def test_missing_and_invalid_values_become_nan():
    df = pd.DataFrame([{"tenure": "abc", "MonthlyCharges": -5, "Contract": "", "SeniorCitizen": 1}])
    X, _ = prepare_features(df)
    assert np.isnan(X["tenure"].iloc[0]) and np.isnan(X["MonthlyCharges"].iloc[0])
    assert pd.isna(X["Contract"].iloc[0])
    assert X["SeniorCitizen"].iloc[0] == "Yes"


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing required"):
        prepare_features(pd.DataFrame([{"tenure": 1}]))


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        prepare_features(pd.DataFrame())


def test_training_data_cleaning():
    X, y, ids = clean_training_data(load_raw_data())
    assert len(X) == len(y) == len(ids) > 7000
    assert set(y.unique()) == {0, 1}
