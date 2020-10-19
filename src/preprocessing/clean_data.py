"""
Cleans the data, using tools from feature_engineering.py
"""

import pandas as pd

from preprocessing.feature_engineering import (
    handle_missing_values,
    feature_engineering,
    split_and_scale,
)


def save(dataframes, paths):
    for df, path in zip(dataframes, paths):
        df.reset_index(drop=True).to_feather(path)


def save_df(fr_dataframes, us_dataframes):
    file_names = [
        "train/X.feather",
        "train/y.feather",
        "test/X.feather",
        "test/y.feather",
    ]
    fr_file_names = ["data/fr/" + file for file in file_names]
    us_file_names = ["data/us/" + file for file in file_names]
    save(fr_dataframes, fr_file_names)
    save(us_dataframes, us_file_names)


def data_cleaning():
    fr_df = pd.read_feather("raw_data/fr_data.feather")
    us_person_df = pd.read_feather("raw_data/us_data_person.feather")
    us_housing_df = pd.read_feather("raw_data/us_data_housing.feather")
    fr_df, us_person_df, us_housing_df = handle_missing_values(
        fr_df, us_person_df, us_housing_df
    )
    fr_df, us_df = feature_engineering(fr_df, us_person_df, us_housing_df)
    fr_dataframes = split_and_scale(fr_df, "fr")
    us_dataframes = split_and_scale(us_df, "us")
    save_df(fr_dataframes, us_dataframes)
