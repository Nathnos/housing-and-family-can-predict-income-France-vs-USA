import os
import gc

import pandas as pd

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
us_columns_housing = ["NP", "TAXP", "FES", "YBL", "NOC", "FPARC", "SERIALNO"]
us_columns_persons = ["POVPIP", "WAGP", "SERIALNO"]


def merged_fr_dataframe():
    fr_dataframes = []
    for file in os.listdir("raw_data/France"):
        fr_dataframes.append(pd.read_csv("raw_data/France/" + file)[fr_columns])
    return pd.concat(fr_dataframes).reset_index(drop=True)


def merged_us_dataframe():
    us_housing_df_a = pd.read_csv(
        "raw_data/USA/ss15husa.csv", usecols=us_columns_housing
    )
    us_housing_df_b = pd.read_csv(
        "raw_data/USA/ss15husb.csv", usecols=us_columns_housing
    )
    us_housing_df = pd.concat([us_housing_df_a, us_housing_df_b])
    us_person_df_a = pd.read_csv(
        "raw_data/USA/ss15pusa.csv", usecols=us_columns_persons
    )
    us_person_df_b = pd.read_csv(
        "raw_data/USA/ss15pusb.csv", usecols=us_columns_persons
    )
    us_person_df = pd.concat([us_person_df_a, us_person_df_b])
    return (
        us_person_df.reset_index(drop=True),
        us_housing_df.reset_index(drop=True),
    )


def csv_to_feather():
    merged_fr_dataframe().to_feather("raw_data/fr_data.feather")
    gc.collect()  # To avoid overloading memory
    us_person_df, us_housing_df = merged_us_dataframe()
    us_person_df.to_feather("raw_data/us_data_person.feather")
    us_housing_df.to_feather("raw_data/us_data_housing.feather")
    check()


def check():
    fr_df = pd.read_feather("raw_data/fr_data.feather")
    us_df_p = pd.read_feather("raw_data/us_data_person.feather")
    us_df_h = pd.read_feather("raw_data/us_data_housing.feather")
    assert us_df_h.shape[1] == len(us_columns_housing)
    assert us_df_p.shape[1] == len(us_columns_persons)
    assert fr_df.shape[1] == len(fr_columns)


if __name__ == "__main__":
    csv_to_feather()
