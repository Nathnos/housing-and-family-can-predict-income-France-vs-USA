# Launch all process again :
```console
python3 -m src.full_processing
```

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
