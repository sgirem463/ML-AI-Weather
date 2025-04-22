# ML-AI-Weather

## Overview

My dataset is from Kaggle: https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data

It is weather data for Australia, contains about 10 years of daily weather observations from many locations across Australia. It's suitable for developing a classifier to predict the next-day rain on the target variable RainTomorrow.

Weather forecasting is very important in modern life, it affects people’s personal and business decisions in many ways. On top of that I have been always interested in knowning what factors affect weather in which way, it's exciting to learn some techniques to do my own weather forecast.

I follow CRISP-DM for this project.

I will build a model for the whole Australia and build separate models for Sydney and Perth, to see how models of the same kind (e.g. LogisticRegression) perform in different datasets.
I will use visualization in for both data analysis and model comparisons, specifically feature correlation heapmap is used to examine the dataset and believed to be quite useful.

The model is also compared to the DummyClassifier and the "common windom" approach by guessing tomorrow's weather is the same as today's.

### Initial Report

This initial report include typical data understanding, data preparation, visualization and a LogisticRegression model.

## Data Understanding

- Examine the dataset with DataFrame functionsd like .info(), .describe(), .sample(), .value_counts()...

## Data Preparation

- Drop entries with NaN value
- Modifiy the column "Date" from value like 2017-06-23 to "06", basically keep the month info.
- OneHot encoding for categorical features
- PolynomialFeature() with degree=2 for numerical features

## Data Analysis

- Use feature correlation heapmap to examine the data
- Calculate majority class ratio, i.e. probability of raining (or not raining) for any day
- Calculate accuracy of "common wisdom" by guessing tomorrow's weather will be the same as today's
- Calculate raining probability on a monthly basis

## Modeling

- Create and fit a DummyClassifier for the whole Australia
- Create and fit a LogisticRegression Classifier for the whole Australia
- Create and fit a LogisticRegression Classifier for Sydney
- Create and fit a LogisticRegression Classifier for Perth
- Create confusion matrix for all these models for comparison
- Create DataFrames for coefficients of the LogisticRegression models, for inteprettion and comparison


## Subsequent Improvements

- Further feature engineering, e.g. craate features of square root values for numerical features
- Create models with DecisionTree, KNN, SVC and ensemble classifiers
- Apply grid search on model hyperperameters
- Additional visualization


The jupyter file is weatherAUS.ipynb
