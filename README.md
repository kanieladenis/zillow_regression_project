# Zillow Regression Project 

## Project Goals

- Create a model that predicts property tax assessed values of single family properties based on 2017 transactions.
- Identify ways to make a better model for Zillow.
- Create new features out of existing features.


## Project Description

### What is the problem?
Zillow needs a better model to predict single home values based on available features. They have requested that I use 2017 data to build a better model.

### Why is it important?
By making better prediction on houses, Zillow can make profits by purchasing undervalued homes

### How will I address it?
I pull five common features to delier an MVP model and build. If there is enough time, the I will add features and/or modeling methods. Delivery will be 

### What will I deliver?
I will deliver a model that beats their model and new features for modeling.


### Initial Questions

- What are the transactions are in 2017?
- What states and counties are the properties located in?
- What is the tax rate per county or fips? (might have to combine columns to calcualate)
- What is the distribution of tax rates for each county?
- What is the distribution of taxes across fips?
- What are the drivers of single family property values?
- Why do some properties have a much higher value than others when they are located so close to each other?
- Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location? 
- Is having 1 bathroom worse than having 2 bedroom

### Data Dictionary

Provded in Repo

### Steps to Reproduce

1. clone my repo. ensure to have all modules.
2. You will need an env.py file that contains the hostname, username and password of the mySQL server that contains the zillow database and tables. Store that env file locally in the repository. 
3. confirm .gitignore is hiding your env.py file so it won't get pushed to GitHub
4. the following libraries are being used: 

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split
    import sklearn.linear_model
    import sklearn.feature_selection
    import sklearn.preprocessing
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import explained_variance_score
    pandas, matplotlib, seaborn, numpy, sklearn and scipy. 

5. the following imports run first for the report

6. you should be able to run telco_classification_final_report.

### The Plan

#### Wrangle

##### Modules (acquire.py + prepare.py + modeling.py + viz.py)

1. write code to acquire data
2. write code to clean data
3. write code to remove NaNs and Outliers
4. write code Explore
5. write code scale
5. write code for feature selection
6. write code modeling
7. write code for evalution
8. After all code runs without error, break into functions
9. create get function with aquire and clean code
10. creaet prep functions with remove NaNs and outliers code
11. create modeling functions scale, feature selection, and modeling code
12. create evaluation functions with rmse code
13 add all fucntionto final report and test

##### Missing Values (report.ipynb)

- removed one 2018 record
- droped columns: transaction_date, parcel_id (no longer needed after data lfiltering)
- replaced pool NaNs with zero
- replaced lot_size NaNs with mean
- replaced year_built NaNs with most used '1955'
- replaced area NaNs with most used '1120'
- dropped columns: garages, neighborhoods (too many NaNs to be usefull)
- dropped left over nulls, < 200

##### Data Split (prepare.py (def function), report.ipynb (run function))

- Use function we have used in class, as that one seems to meet all the requirements. 

##### Using your modules (report.ipynb)

- created acquire, prepare, modeling, and viz modules. 

#### Explore

##### Ask a clear question, [discover], provide a clear answer (report.ipynb)

- What are the transactions are in 2017?
- What states and counties are the properties located in?
- What is the tax rate per county or fips? (might have to combine columns to calcualate)
- What is the distribution of tax rates for each county?
- What is the distribution of taxes across fips?
- What are the drivers of single family property values?
- Why do some properties have a much higher value than others when they are located so close to each other?
- Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location? 
- Is having 1 bathroom worse than having 2 bedrooms?


##### Exploring through visualizations (report.ipynb)


##### Statistical tests (report.ipynb)

- I ran a two sample - one tail T Test ot find if Orange County values were greater than Los Angeles County value averages
- I ran a one sample - one tail T Test to find if Orange County values were great than all counties mean

##### Summary (report.ipynb)

- We have 47,500 houses across three California counites of (Los Angeles, Ventura, and Orange)
- Best Three drivers are Area, Bathrooms, and Bedrooms in that order
- Los Angels County has a .2% higher tax rate than the other counties. It also had significantly more houses with a value distribution that peaks between 100-300K.
- Ventura and Orange Counties have a lower tax rate and with significantly less houses that have a value distribution that peak between 300-500K.
- More high value hosues are along the coast and in Orange County

#### Modeling

##### Select Evaluation Metric (Report.ipynb)

RMSE

##### Evaluate Baseline (Report.ipynb)

Baseline set for mean of actual values

##### Develop 3 Models (Report.ipynb)

- created OLS, LassLars, TWeedieRegressor, Polynomial

#### Evaluate on Train (Report.ipynb)

All models RMSE beat baseline, Polynomial did best.

##### Evaluate on Validate (Report.ipynb)
 
All models RMSE beat baseline, Polynomial did best.

##### Evaluate Top Model on Test (Report.ipynb)

Polynomial performed bad. need to chek data split.

