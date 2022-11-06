In this document we write down the phases of this project!

# Sections

- [Logging](#Logging)
- [Loading](#Loading)
- [EDA](#EDA)
- [PreProcessing](#PreProcessing)
- [Dimensionality Reduction](#Dimensionality-reduction)
- [Models Selection](#Model-Selection)
- [Model Deployment](#Model-Deployment)
- [Optimization Techniques](#Optimization-Techniques)


## Logging
In this phase we create a logging mechanism using the python built in logging module in order to log into a file possible errors, warnings or information. 

## Loading
This is the first phase of our project in which we include all the steps regarding the import of our data. Firsly, we read the csv file from the Kaggle source and then we apply some data cleaning techniques (validation checks, errors, inconsistencies, handling missing data) to prepare the data for the next steps.

## EDA
This section is called Exploratory Data Analysis. We create some plots to have a deeper understanding of the distribution of our data in both categorical and numeircal features.

## PreProcessing
This phase encaplulates all the pre-processing steps that are needed for this project such as one-hot encoding techinque for categorical data, feature scaling and data spliting into train and test set.

## Dimensionality Reduction
This phase provides some unsupervised machine learning algorithms such as PCA and K-means. In this phase we also create some plots to have a crystal-clear point of what these algorithms can do to our data.

## Models Selection
In this phase we evaluate several models so we can obtain the best resutls possible in terms of accuracy. We focus on supervised machine learning regression algorithms.

## Model Deployment
This section provides the models which had the best performance from the previous step. We utlise the following machine learingn algorithms: RandomForest and XGB.

## Optimization Techniques
In the last step, we apply some optimization techniques using numpy library and profiling to optimize our analysis in terms of computational time without decreasing the accuracy of the models.
