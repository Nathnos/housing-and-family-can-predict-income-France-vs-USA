import os
import gc

import pandas as pd

"""
Convert and merge multiple CSV files into a unique feather file
"""

# France :
fr_dataframes = []
for file in os.listdir("raw_data/France"):
    fr_dataframes.append(pd.read_csv("raw_data/France/" + file))
pd.concat(fr_dataframes).reset_index().to_feather("raw_data/fr_data.feather")
gc.collect()  # To avoid overloading memory

# USA :
us_dataframes = []
for file in os.listdir("raw_data/USA"):
    us_dataframes.append(pd.read_csv("raw_data/USA/" + file))
pd.concat(us_dataframes).reset_index().to_feather("raw_data/us_data.feather")
