# Launch all process again :
```console
python3 -m src.full_processing
```

# Get the data
I found my data on [census.gouv](https://www.census.gov/programs-surveys/acs/microdata/access.2015.html) and [insee.fr](https://www.insee.fr/fr/statistiques/4176293).

The french dataset gives the mean values for a specific eara.  
The US dataset gives values for every housing.

# Choose variables
Using the data dictionary from both data sets, I managed to found the name of relevant features (average values for France) : 


## Input Features :
- NP : Number of people in a Housing
- H1 : Housing with only one people
- H5 : Housing with 5 people or more
- SPH : Single-Parent Housing
- HY : Year of House Construction
- OH : Own House

## Output Features :
- PP : Proportion of Poor People (FR) / Is housing Poor (US)
- SL : Standard of Living ([french calculation](https://fr.wikipedia.org/wiki/Niveau_de_vie_en_France))

# Convert the data
Let's now merge CSV files into a unique [feather](https://github.com/wesm/feather) file.  
I loaded every csv file, selected relevent columns, merged all dataframes, then saved to feather.  
I then handled missing values, and did some Feature engineering.

For France, instead of predicting the porportion of poor people, the model predicts a class representing an interval.

![classes graph](https://raw.githubusercontent.com/Nathnos/housing-and-family-can-predict-income-France-vs-USA/master/medias/french_classes.jpg)

For a deeper look, you can check the preprocessing folder.
