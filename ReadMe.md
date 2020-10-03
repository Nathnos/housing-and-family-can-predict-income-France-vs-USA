# Get the data
Data for France is quite simple to get, as there is in only one zip file to download, and three csv files in total. But it was a bit hard for the USA data, since there is one file per state.

I found the FTP of the Census Bureau (where I get my data from), and I downloaded the files I wanted with wget :
>wget -r -nd -A "csv_hc*.zip" -P raw_data/USA ftp://ftp2.census.gov/programs-surveys/acs/data/pums/2015/1-Year/

I then unziped all the files and removed the .pdf information files to keep only the csv files.
>cd raw_data/USA<br/>
>unzip -n "*.zip" && rm *.{zip,pdf}

# Convert the data
Let's now merge CSV files into a unique feather file. Since we have too much USA data for what is needed, I decided to remove the 10 larger files, in order to make sure not to overload RAM.
>du -a raw_data/USA/ | sort -n -r | tail -n +2 | head -n 10 | cut -d $'\t' -f2 | xargs rm


```python
import os
import gc
import pandas as pd

#France :
fr_dataframes = []
for file in os.listdir("raw_data/France"):
    fr_dataframes.append(pd.read_csv("raw_data/France/" + file))
pd.concat(fr_dataframes).reset_index().to_feather("raw_data/fr_data.feather") #Save to feather format
gc.collect() #To avoid overloading memory

#USA :
us_dataframes = []
for file in os.listdir("raw_data/USA"):
    us_dataframes.append(pd.read_csv("raw_data/USA/" + file))
pd.concat(us_dataframes).reset_index().to_feather("raw_data/us_data.feather")
```

# Choosing variables
…
