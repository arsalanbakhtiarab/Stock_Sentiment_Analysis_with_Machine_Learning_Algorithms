# Stock Sentiment Analysis with Machine Learning Algorithms

## Overview

This repository contains a Python script for performing sentiment analysis on stock news using various machine learning algorithms. The dataset used for this analysis is sourced from Kaggle and consists of headlines related to stock market news. The goal of this project is to predict stock sentiment (positive or negative) based on the provided news headlines.

## Dataset

The dataset used for this project can be found on Kaggle. It includes news headlines and corresponding labels indicating whether the news had a positive or negative impact on the stock market. The headlines are preprocessed to ensure better results during the analysis.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python (3.7 or higher)
- pandas
- numpy
- scikit-learn
- nltk
- seaborn (for visualization)

You can install the required packages using pip:

```
pip install pandas numpy scikit-learn nltk seaborn
```

## How to Use

1. Clone this repository to your local machine.
2. Make sure you have the required Python packages installed (see Prerequisites).
3. Download the dataset from the provided Kaggle link and place it in the appropriate location or update the data file path in the script accordingly.
4. Run the Python script (main.py).

## Script Workflow

1. Data Loading: The script reads the CSV file containing the stock news headlines and their labels.
2. Data Preprocessing: The headlines are preprocessed to remove punctuation, convert text to lowercase, and tokenize the words. Stop words are also removed, and words are lemmatized to their root form.
3. Feature Extraction: The script implements two feature extraction techniques - Bag of Words and TF-IDF.
4. Model Training: Several machine learning classifiers (Multinomial Naive Bayes, Passive Aggressive Classifier, Random Forest, and Decision Tree) are trained using the feature vectors obtained from the chosen feature extraction method.
5. Model Evaluation: The accuracy of each classifier is evaluated using a test dataset, and a confusion matrix is generated for further analysis.
6. Model Comparison: The performance of different classifiers is compared in terms of accuracy, and a visualization is provided to aid in model selection.

## Results

The script displays the accuracy achieved by each classifier and generates confusion matrices for further analysis. Based on the results, you can select the most appropriate model for your sentiment analysis task.

## Contributions

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, feel free to open a pull request or create an issue.

## About the Author

This project is maintained by [Arsalan Bakhtiar AB](https://github.com/arsalanbakhtiarab). Connect with me on LinkedIn [here](https://www.linkedin.com/in/arsalan-bakhtiar-ab-1b8467253/).

Happy coding!
