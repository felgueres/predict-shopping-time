#### Modeling Shopping Time
#### Date: July 28th, 2017

#### Description:

Modeling shopping time duration for inferece, includes:

1) understand the data via exploration (EDA),
2) design a workflow to transform raw data into the feature space of the model,
3) build model and predict
4) **extra**: inference of resulting feature space and model

#### Model:

- L1 Regularized Regression (Lasso)
- Random Forest (minor exploration)

#### Code:

> preprocessing.py - contains Preprocessing class to:

- Preprocessing of data prior to model fit
- Returns feature engineered datasets
- Handles dummy variables without leakages
- Handles scaling

> model.py

- Loads data
- Featurizes data
- Runs Lasso Model
- Outputs predictions into folder ./predictions/

#### File usage for predictions of the test-set:  

- In console, navigate to ./src/ folder.
- run -model.py

#### Used Technologies:

- Python
- Pandas
- Matplotlib / Seaborn
- sklearn
- StatsModels
- Jupyter Notebook
