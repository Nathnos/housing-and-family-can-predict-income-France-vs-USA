# Get the data
I found my data on [census.gouv](https://www.census.gov/programs-surveys/acs/microdata/access.2015.html) and [insee.fr](https://www.insee.fr/fr/statistiques/4176293)

# Choose variables
Using the data dictionary from both data sets, I managed to found the name of relevant features : 


## Input Features :
- NP : Average Number of people in a Housing
- H1 : Proportion of Housing with only one people
- H5 : Proportion of Housing with 5 people or more
- SPH : Proportion of Single-Parent Housing
- HY : Average Year of House Construction
- HO : Proportion of House Owners

## Output Features (to predict) :
- PP : Proportion of Poor People
- SL : Average Standard of Living ([french calculation](https://fr.wikipedia.org/wiki/Niveau_de_vie_en_France))

# Convert the data
Let's now merge CSV files into a unique [feather](https://github.com/wesm/feather) file.
I loaded every csv file, selected relevent columns, merged all dataframes, then saved to feather.
I then handled missing values, and did some Feature engineering.

The there is about 13% of poor people, so that's gonna be the accuracy on random models. That sets a comparison point for other models.

For a deeper look, you can check the preprocessing folder.


```python
import pandas as pd
import numpy as np

us_person_df = pd.read_feather("raw_data/us_data_person.feather")
us_housing_df = pd.read_feather("raw_data/us_data_housing.feather")

def count_cu(NP, NbC):
    NP -= 1
    remaining_ad = NP-NbC
    return 1 + remaining_ad*0.5 + NbC*0.3

i=50
#us_person_df["POVPIP"] = us_person_df.groupby("SERIALNO").max()["POVPIP"].reset_index(drop=True)
#us_person_df["WAGP"] = us_person_df.groupby("SERIALNO").sum()["WAGP"].reset_index(drop=True)
df_povpip = us_person_df[["SERIALNO", "POVPIP"]].groupby("SERIALNO").max()
df_wagp = us_person_df[["SERIALNO", "WAGP"]].groupby("SERIALNO").sum()
us_person_df = df_wagp.merge(df_povpip, on="SERIALNO")
us_df = us_person_df.join(us_housing_df, on="SERIALNO")
us_df = us_df[us_df["NP"] >= 1]
#us_df["CU"] = us_df.apply(lambda row: count_cu(row["NP"], row["NbC"]), axis=1)
#us_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAGP</th>
      <th>POVPIP</th>
      <th>SERIALNO</th>
      <th>NP</th>
      <th>YBL</th>
      <th>FES</th>
      <th>FPARC</th>
      <th>NOC</th>
      <th>TAXP</th>
      <th>NbC</th>
      <th>CU</th>
    </tr>
    <tr>
      <th>SERIALNO</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>160750.0</td>
      <td>501.0</td>
      <td>67.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>103.0</td>
      <td>160.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70000.0</td>
      <td>441.0</td>
      <td>345.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>135000.0</td>
      <td>468.0</td>
      <td>463.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>80000.0</td>
      <td>501.0</td>
      <td>518.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1496667</th>
      <td>22200.0</td>
      <td>265.0</td>
      <td>1507885.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>0</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1496669</th>
      <td>0.0</td>
      <td>218.0</td>
      <td>1509341.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1496670</th>
      <td>65000.0</td>
      <td>410.0</td>
      <td>1509548.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1496675</th>
      <td>82000.0</td>
      <td>501.0</td>
      <td>1511171.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1496676</th>
      <td>0.0</td>
      <td>501.0</td>
      <td>1512206.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
<p>1250288 rows × 11 columns</p>
</div>




```python
fr_dataframe = pd.read_feather("raw_data/fr_data.feather")
fr_dataframe["NP"] = fr_dataframe["Ind"] / fr_dataframe["Men"]
fr_dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ind</th>
      <th>Men</th>
      <th>Men_prop</th>
      <th>Men_1ind</th>
      <th>Men_5ind</th>
      <th>Men_fmp</th>
      <th>Log_av45</th>
      <th>Log_45_70</th>
      <th>Log_70_90</th>
      <th>Log_ap90</th>
      <th>Men_pauv</th>
      <th>Ind_snv</th>
      <th>NP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>2.4</td>
      <td>1.2</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.4</td>
      <td>130176.6</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.0</td>
      <td>7.8</td>
      <td>3.8</td>
      <td>1.2</td>
      <td>0.3</td>
      <td>1.2</td>
      <td>0.3</td>
      <td>0.6</td>
      <td>2.1</td>
      <td>2.6</td>
      <td>1.2</td>
      <td>412225.9</td>
      <td>2.435897</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.0</td>
      <td>7.1</td>
      <td>3.9</td>
      <td>2.6</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.2</td>
      <td>1.8</td>
      <td>371563.8</td>
      <td>2.253521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>2.6</td>
      <td>1.4</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>1.6</td>
      <td>0.7</td>
      <td>139336.4</td>
      <td>2.307692</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.0</td>
      <td>4.8</td>
      <td>2.6</td>
      <td>1.8</td>
      <td>0.4</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.2</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>255450.1</td>
      <td>2.291667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>377229</th>
      <td>985.0</td>
      <td>366.0</td>
      <td>55.9</td>
      <td>110.0</td>
      <td>41.0</td>
      <td>84.0</td>
      <td>2.9</td>
      <td>87.1</td>
      <td>14.0</td>
      <td>243.0</td>
      <td>130.0</td>
      <td>16601889.9</td>
      <td>2.691257</td>
    </tr>
    <tr>
      <th>377230</th>
      <td>5030.5</td>
      <td>2088.0</td>
      <td>427.0</td>
      <td>745.0</td>
      <td>207.0</td>
      <td>437.0</td>
      <td>86.0</td>
      <td>158.0</td>
      <td>282.1</td>
      <td>1550.9</td>
      <td>567.0</td>
      <td>107644141.6</td>
      <td>2.409243</td>
    </tr>
    <tr>
      <th>377231</th>
      <td>5693.0</td>
      <td>2242.0</td>
      <td>471.0</td>
      <td>710.0</td>
      <td>244.0</td>
      <td>439.0</td>
      <td>77.0</td>
      <td>139.9</td>
      <td>210.0</td>
      <td>1794.1</td>
      <td>783.0</td>
      <td>111375421.9</td>
      <td>2.539251</td>
    </tr>
    <tr>
      <th>377232</th>
      <td>77.0</td>
      <td>32.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>14.9</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>1185450.8</td>
      <td>2.406250</td>
    </tr>
    <tr>
      <th>377233</th>
      <td>380.0</td>
      <td>169.0</td>
      <td>62.0</td>
      <td>73.0</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>37.0</td>
      <td>98.0</td>
      <td>38.0</td>
      <td>10002946.8</td>
      <td>2.248521</td>
    </tr>
  </tbody>
</table>
<p>377234 rows × 13 columns</p>
</div>




```python
fr_dataframe = pd.read_feather("data/us/train/y.feather")
fr_dataframe["PP"].mean()
```




    0.12505755190633838


