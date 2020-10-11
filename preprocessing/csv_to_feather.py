import os
import gc

import pandas as pd
from pickle import dump

"""
Convert and merge multiple CSV files into a unique feather file
"""

fr_columns = [
    "Ind",
    "Men",
    "Men_prop",
    "Men_1ind",
    "Men_5ind",
    "Men_fmp",
    "Log_av45",
    "Log_45_70",
    "Log_70_90",
    "Log_ap90",
    "Men_pauv",
    "Ind_snv",
]
us_columns = ["NP", "TAXP", "FES", "YBL", "POVPIP", "WAGP", "NOC", "FPARC"]


def merged_fr_dataframe():
    fr_dataframes = []
    for file in os.listdir("raw_data/France"):
        fr_dataframes.append(pd.read_csv("raw_data/France/" + file)[fr_columns])
    return pd.concat(fr_dataframes).reset_index().iloc[:, 1:]


def merged_us_dataframe():
    us_housing_df_a = pd.read_csv("raw_data/USA/ss15husa.csv")[us_columns]
    us_housing_df_b = pd.read_csv("raw_data/USA/ss15husa.csv")[us_columns]
    us_housing_df = pd.concat(us_housing_df_a, us_housing_df_b)
    us_person_df_a = pd.read_csv("raw_data/USA/ss15pusa.csv")[us_columns]
    us_person_df_b = pd.read_csv("raw_data/USA/ss15pusb.csv")[us_columns]
    us_person_df = pd.concat(us_person_df_a, us_person_df_b)
    return pd.merge(us_housing_df, us_person_df).reset_index().iloc[:, 1:]


def csv_to_feather():
    merged_fr_dataframe.to_feather("raw_data/fr_data.feather")
    gc.collect()  # To avoid overloading memory
    merged_us_dataframe.to_feather("raw_data/us_data.feather")
    fr_df = pd.read_feather("raw_data/us_data.feather")
    us_df = pd.read_feather("raw_data/us_data.feather")
    assert us_df.shape[1] == len(us_columns)
    assert fr_df.shape[1] == len(fr_columns)


if __name__ == "__main__":
    csv_to_feather()
