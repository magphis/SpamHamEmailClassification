# Spam-Ham Email Classification Project
This repository contains a machine learning project focused on classifying emails as either spam or ham (non-spam). The project utilizes the SpamAssassin dataset and implements three different machine learning models from scratch using only NumPy. Additionally, a customized data preprocessing script is provided to clean and preprocess the email data before feeding it into the models.

## Dataset
The dataset used in this project is obtained from SpamAssassin.

## Machine Learning Models
Three different machine learning models have been implemented from scratch using NumPy:

**BernoulliNBScratch.py**: Implementation of a Bernoulli Naive Bayes classifier for email classification.
**LogisticRegressionScratch.py**: Implementation of Logistic Regression for email classification.
**MultinomialNBScratch.py**: Implementation of a Multinomial Naive Bayes classifier for email classification.

## Data Preprocessing
The preprocess.py script is provided to perform data cleaning and preprocessing on the email data. It includes steps such as tokenization, lowercasing, removing stop words, and converting text data into a format suitable for the machine learning models.

## Jupyter Notebook
The SpamHamClassification.ipynb Jupyter Notebook showcases the complete implementation of the project. It covers data loading, preprocessing, model training, evaluation, and visualization of results. This notebook serves as a comprehensive guide to understanding and reproducing the entire project.

## Contributions
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
