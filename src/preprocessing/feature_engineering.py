"""
Some feature engineering :
-Handle missing values
-Get or create relevant columns for both datasets
-Scale input (for a better learning)
-Split data into train and test sets.
"""

from pickle import dump

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

input_features = ["NP", "H1", "H5", "SPH", "HY", "OH"]
output_features = ["PP", "SL"]
features = input_features + output_features

us_YBL_converter = {
    1: 1939,
    2: 1945,
    3: 1955,
    4: 1965,
    5: 1975,
    6: 1985,
    7: 1995,
    8: 2002,
    9: 2005,
    10: 2006,
    11: 2007,
    12: 2008,
    13: 2009,
    14: 20010,
    15: 2011,
    16: 2012,
    17: 2013,
}


def handle_missing_values(fr_dataframe, us_person_df, us_housing_df):
    # Drop lines with no prediction
    us_person_df = us_person_df[us_person_df["POVPIP"].notna()]
    us_person_df = us_person_df[us_person_df["WAGP"].notna()]
    us_person_df = us_person_df[us_person_df["WAGP"] > 0]
    fr_dataframe = fr_dataframe[fr_dataframe["Men_pauv"].notna()]
    fr_dataframe = fr_dataframe[fr_dataframe["Ind_snv"].notna()]
    # For input data, fill empty values with median
    fr_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    us_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_fr_dataframe = pd.DataFrame(fr_imputer.fit_transform(fr_dataframe))
    imp_fr_dataframe.columns = fr_dataframe.columns
    imp_fr_dataframe.index = fr_dataframe.index
    imp_us_housing_df = pd.DataFrame(us_imputer.fit_transform(us_housing_df))
    imp_us_housing_df.columns = us_housing_df.columns
    imp_us_housing_df.index = us_housing_df.index
    return fr_dataframe, us_person_df, us_housing_df


def shuffle(dataframe):
    return dataframe.sample(frac=1).reset_index(drop=True)


def split_and_scale(dataframe, directory):
    dataframe = shuffle(dataframe)
    X = dataframe[input_features]
    y = dataframe[output_features]
    scaler = StandardScaler().fit(X)
    path = "data/" + directory + "/scaler.pkl"
    dump(scaler, open(path, "wb"))
    X = scaler.transform(X)
    test_proportion = 0.2
    test_set_size = int(test_proportion * len(dataframe))
    y_test, y_train = y[:test_set_size], y[test_set_size:]
    X_test, X_train = X[:test_set_size], X[test_set_size:]
    X_test = pd.DataFrame(X_test, columns=dataframe[input_features].columns)
    X_train = pd.DataFrame(X_train, columns=dataframe[input_features].columns)
    y_test = pd.DataFrame(y_test, columns=dataframe[output_features].columns)
    y_train = pd.DataFrame(y_train, columns=dataframe[output_features].columns)
    return X_train, y_train, X_test, y_test


def feature_engineering(fr_dataframe, us_person_df, us_housing_df):
    return (
        fr_engineering(fr_dataframe),
        us_engineering(us_person_df, us_housing_df),
    )


def us_engineering(us_person_df, us_housing_df):
    # Merge person and housing dataframes
    df_povpip = us_person_df[["SERIALNO", "POVPIP"]].groupby("SERIALNO").max()
    df_wagp = us_person_df[["SERIALNO", "WAGP"]].groupby("SERIALNO").sum()
    us_person_df = df_wagp.merge(df_povpip, on="SERIALNO")
    us_dataframe = us_person_df.join(us_housing_df)
    # Drop empty houses
    us_dataframe = us_dataframe[us_dataframe["NP"] >= 1]
    us_dataframe["YBL"].replace(us_YBL_converter, inplace=True)
    us_dataframe["HY"] = us_dataframe["YBL"]
    us_dataframe["H1"] = (us_dataframe["NP"] == 1).astype(int)
    us_dataframe["H5"] = (us_dataframe["NP"] >= 5).astype(int)
    us_dataframe["SPH"] = (us_dataframe["NP"] >= 5).astype(int)
    us_dataframe["OH"] = (us_dataframe["TAXP"] > 1).astype(int)
    us_dataframe["PP"] = (us_dataframe["POVPIP"] < 100).astype(int)
    # CU stands for Consumption Units ; used for Standard of living
    us_dataframe["CU"] = us_dataframe.apply(count_cu, axis=1)
    us_dataframe = us_dataframe[us_dataframe["CU"] > 0]
    us_dataframe["SL"] = us_dataframe["WAGP"] / us_dataframe["CU"]
    return us_dataframe[features]


def count_cu(row):
    number_of_person = row["NOC"]
    number_of_children = row["NP"]
    number_of_person -= 1
    remaining_adults = max(number_of_person - number_of_children, 0)
    return 1 + remaining_adults * 0.5 + number_of_children * 0.3


def fr_engineering(fr_dataframe):
    fr_dataframe["NP"] = fr_dataframe["Ind"] / fr_dataframe["Men"]
    fr_dataframe["H1"] = fr_dataframe["Men_1ind"] / fr_dataframe["Men"]
    fr_dataframe["H5"] = fr_dataframe["Men_5ind"] / fr_dataframe["Men"]
    fr_dataframe["SPH"] = fr_dataframe["Men_fmp"] / fr_dataframe["Men"]
    df_weighted_sum = (
            fr_dataframe["Log_av45"] * 1945
            + fr_dataframe["Log_45_70"] * 1958
            + fr_dataframe["Log_70_90"] * 1980
            + fr_dataframe["Log_ap90"] * 1995
    )
    df_sum = (
            fr_dataframe["Log_av45"]
            + fr_dataframe["Log_45_70"]
            + fr_dataframe["Log_70_90"]
            + fr_dataframe["Log_ap90"]
    )
    fr_dataframe["HY"] = df_weighted_sum / df_sum
    fr_dataframe["OH"] = (
            fr_dataframe["Men_prop"].astype("float") / fr_dataframe["Men"]
    )
    fr_dataframe["PP"] = (
            fr_dataframe["Men_pauv"].astype("float") / fr_dataframe["Men"]
    )
    fr_dataframe["PP"] = fr_dataframe.apply(french_pp_class, axis=1)
    fr_dataframe["SL"] = (
            fr_dataframe["Ind_snv"].astype("float") / fr_dataframe["Ind"]
    )
    return fr_dataframe[features]


def french_pp_class(row):
    """
    Transforms a proportion of poor people (pp) into a class representing an interval of proportion.
    """
    poor_people_proportion = row["PP"]
    if poor_people_proportion < 0.05:
        return 0
    if poor_people_proportion < 0.09:
        return 1
    if poor_people_proportion < 0.13:
        return 2
    if poor_people_proportion < 0.18:
        return 3
    if poor_people_proportion < 0.25:
        return 4
    return 5
