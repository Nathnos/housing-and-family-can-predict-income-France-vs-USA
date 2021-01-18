# Caution !
This is an outdated project, and I'm not even sure I read the data correctly, so be careful.

# France VS USA: How Accurately Can Salary Be Predicted By Housing And Family Data ?

This project aims at knowing if some housing properties and family status were good predictors of income. 

## Results

### What is similar
I noticed some global trends between these two states, but surprisingly a few only :
- There is a larger proportion of 5+ people living together in poor housings.
- Unsuprisingly, being a single parent is a good predictor of poverty.

However, these trends are exacerbated in us data.

![SPH similarites](https://raw.githubusercontent.com/Nathnos/housing-and-family-can-predict-income-France-vs-USA/master/medias/SPH_similarites.jpg)

### What is diffrent
I found some interesting diffrences between France and US data.
- Rich housings tends to have more people in US and less in France.
- Rich American people live more often alone than rich French people.
- In France, poor people have older houses and don't own them. In US, there are no diffrences.

![Housing differences](https://raw.githubusercontent.com/Nathnos/housing-and-family-can-predict-income-France-vs-USA/master/medias/housing_differences.jpg)

### Predicting neighborhood wealth
I tried to predict the wealth of a 1km Housing agglomeration. I encoded data to predict as 6 classes (see below), so a random model gets 16.7% accuracy. I trained a Deep Neural Network who get 32% accuracy.

In conclusion, there is a link between housing properties and wealth, but it's not enought to be precise.

# Cleaning the data
I had to face several problems :
- Too many variables : 439 for US raw data; only 8 relevant
- Diffrent variable for France of US
- Datasets were a bit messy

![Data cleaning](https://raw.githubusercontent.com/Nathnos/housing-and-family-can-predict-income-France-vs-USA/master/medias/cleaning.jpg)

# Launch all process again :
```console
python3 -m src.full_processing
```

# Get the data
I found my data on [census.gouv](https://www.census.gov/programs-surveys/acs/microdata/access.2015.html) and [insee.fr](https://www.insee.fr/fr/statistiques/4176293).

The french dataset gives the mean values for a specific eara.  
The US dataset gives values for every housing.

# Variable Choice
Using the data dictionary from both data sets, I managed to found the name of relevant features (average values for France) : 


## Input Features / Predictors:
- NP : Number of people in a Housing
- H1 : Housing with only one people
- H5 : Housing with 5 people or more
- SPH : Single-Parent Housing
- HY : Year of House Construction
- OH : Own House

## Output Features :
- PP : Proportion of Poor People (FR) / Is housing Poor (US). Encoded as classes (see below).
- SL : Standard of Living ([french calculation](https://fr.wikipedia.org/wiki/Niveau_de_vie_en_France))

# Convert the data
Let's now merge CSV files into a unique [feather](https://github.com/wesm/feather) file.  
I loaded every csv file, selected relevent columns, merged all dataframes, then saved to feather.  
I then handled missing values, and did some Feature engineering.

For France, instead of predicting the porportion of poor people, the model predicts a class representing an interval. Every interval has about the same amount of data.

![classes graph](https://raw.githubusercontent.com/Nathnos/housing-and-family-can-predict-income-France-vs-USA/master/medias/french_classes.jpg)

For US, there are three classes : people with less than poverty, people with 5+ times poverty threshold, and people in between.

Yet, the Standard of Living is a much better way to compare the two states, as both countries have diffrent ways to set poverty threshold. 

For a deeper look, you can check the preprocessing folder.
