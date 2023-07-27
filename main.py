# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:37:05 2022

@author: Arsalan Bakhtiar
"""

# Data Set link = https://www.kaggle.com/arran7sun/stocknews

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


df = pd.read_csv(r'D:\Source Code\Stock Sentiment Analysis\Data.csv',
                 encoding=('ISO-8859-1'),parse_dates=['Date'])

df.columns
df.tail()
df.fillna(value='who',inplace=True)


#Creating new indexing
new_index = [str(i) for i in range(27)]
df.columns = new_index

#Seperating the data and label
data = df.iloc[:,2:27]
label = df['1']


# Removing puncation
# data = data.iloc[:,2:27] #All  columns : , and row form 2 : to 27 index
data.replace('[^a-zA-Z]'," ",regex = True, inplace = True)

# Converting headlines to lower case
for index in range(2,27):
    data[str(index)] = data[str(index)].str.lower()


# Join all the deadlines 
headlines = [] 
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

#Tokenization
Words_Tokenize = []
for index in range(0,len(data.index)):
    Words_Tokenize.append(word_tokenize(headlines[index],language='english'))

#Stop Words set
stop_words = set(stopwords.words('english'))


# Implementing the Lemmatizer & POS tag 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer  = WordNetLemmatizer()

POS_Tag = []
for index in range(0,len(data.index)):
    POS_Tag.append(nltk.pos_tag(Words_Tokenize[index]))
        

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatizing
Final_Lemmatize = []
for index in range(0,len(data.index)):
    sw = []
    for word,tag in POS_Tag[index]:
        lemma = lemmatizer.lemmatize(word,pos=get_pos(tag))
        sw.append(lemma)
    Final_Lemmatize.append(sw)


#Removing Stop Words
Stop_Words = []
for index in range(0,len(data.index)):
    Stop_Words.append(str([w for w in Final_Lemmatize[index] if not w in stop_words]))


Finalized_Data = Stop_Words.copy()


# Applaying Countvectorization
# Creating the Bag of Words
#==============================
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Assuming Finalized_Data and df are defined and contain appropriate data

# Create a CountVectorizer instance
cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))

# Transform the text data into a numerical format
X = cv.fit_transform(Finalized_Data).toarray()

# Assign the target variable
y = df['1']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Access the feature names from the CountVectorizer
feature_names = cv.get_feature_names_out()

# Create a DataFrame from the training data to visualize the count of features
count_df = pd.DataFrame(X_train, columns=feature_names)


# Implemanting MultinomialNB
# ==========================
from sklearn.naive_bayes import MultinomialNB

# Create an instance of MultinomialNB classifier
classifier = MultinomialNB()

from sklearn import metrics

# Train the classifier using the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (Bag of Words) and (MultinomialNB ): %0.3f" % score)


# Calculate the confusion matrix
cm = metrics.confusion_matrix(y_test, pred)


# Confusion Matrix  of Multionial NB
# ==================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix MultinomialNB (Bag of Words)')
plt.show()


# Implemanting Passive Aggressive Classifier Algorithm
# ====================================================

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter_no_change = 50)

linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (Bag of Words) and (PassiveAggressiveClassifier): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)

# Confusion Matrix  Passive Aggressive Classifier Algorithm
# =======================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Passive Agressive Classifier Algo. (Bag of Words)')
plt.show()


# Implemanting Random Forest Classifier
# ====================================

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
pred = rf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (Bag of Words) and (Random Forest): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)

# Confusion Matrix  Random Forest Classifier
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Random Forest Classifier (Bag of Words)')
plt.show()



# Implemanting Decision Tree
# ==========================

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X, y)
pred = dt.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (Bag of Words) and (Decision Tree): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)


# Confusion Matrix  Decision Tree
# ===============================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Decision Tree (Bag of Words)')
plt.show()

# Comparision of Model 
# ====================

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Multinomial NB': MultinomialNB(),
    'Passive Agg.Clas.':PassiveAggressiveClassifier(),
}

# Train and evaluate each classifier
accuracies = {}
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracies[clf_name] = accuracy_score(y_test, pred)

# Plot the accuracies of different classifiers
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies (Bag of Words)')
plt.xticks(rotation=45)
plt.show()



# Implement the TF-IDF
#=====================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
x = tfidf_v.fit_transform(Finalized_Data).toarray()
y = df['1']

# Create a CountVectorizer instance
cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))

# Transform the text data into a numerical format
X = cv.fit_transform(Finalized_Data).toarray()

# Assign the target variable
y = df['1']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Access the feature names from the CountVectorizer
feature_names = cv.get_feature_names_out()

# Create a DataFrame from the training data to visualize the count of features
count_df = pd.DataFrame(X_train, columns=feature_names)


# Implemanting MultinomialNB
# ==========================
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

from sklearn import metrics

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (TF-IDF) and (MultinomialNB ): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)

# Confusion Matrix  Passive Aggressive Classifier Algorithm
# =======================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix MultinomialNB (TF-IDF)')
plt.show()


# Implemanting Passive Aggressive Classifier Algorithm
# ====================================================

from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter_no_change = 50)

linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (TF-IDF) and (PassiveAggressiveClassifier): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)



# Confusion Matrix  Passive Aggressive Classifier Algorithm
# =======================================================

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Passive Agressive Classifier Algo. (TF-IDF)')
plt.show()


# Implemanting Random Forest Classifier
# ====================================

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
pred = rf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (Bag of Words) and (Random Forest): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)


# Confusion Matrix  Random Forest Classifier
# ==========================================

import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Random Forest Classifier (TF-IDF)')
plt.show()


# Implemanting Decision Tree
# ==========================

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X, y)
pred = dt.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print("\nAccuracy With (Bag of Words) and (Decision Tree): %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)

# Confusion Matrix  Decision Tree
# ===============================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Decision Tree (TF_IDF)')
plt.show()


# Comparision of Model 
# ====================

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Multinomial NB': MultinomialNB(),
    'Passive Agg.Clas.':PassiveAggressiveClassifier(),
}

# Train and evaluate each classifier
accuracies = {}
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracies[clf_name] = accuracy_score(y_test, pred)

# Plot the accuracies of different classifiers
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies (TF-IDF)')
plt.xticks(rotation=45)
plt.show()


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have the feature data 'X' and the target labels 'y'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Hyperparameter values (maximum depth) to try
max_depth_values = [1, 2, 3, 4, 5]

# Lists to store accuracy values for each maximum depth value
accuracies = []

# Train and evaluate the classifier for each maximum depth value
for max_depth in max_depth_values:
    # Initialize the Decision Tree classifier with the current maximum depth
    dt = DecisionTreeClassifier(max_depth=max_depth)

    # Train the classifier
    dt.fit(X_train, y_train)

    # Make predictions on the test set
    pred = dt.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, pred)
    accuracies.append(accuracy)

# Plot the model performance as a line plot
plt.plot(max_depth_values, accuracies, marker='o', linestyle='-')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Model Performance of Decision Tree Classifier')
plt.show()
