# Juan Davalos and Michael Williamson
# Computer Science 542 - Machine Learning
# Good and Evil Classifier Code

# Natural Language Processing
# Sentiment Analysis

# Importing the libraries
import numpy as np
import pandas as pd
from pathlib import Path

# Importing the dataset
path = Path(__file__).parent / 'Good_Evil - Sheet1.tsv' 
ge_dataset = pd.read_csv(path, delimiter = '\t', quoting = 3)

path = Path(__file__).parent / 'Lawful_Chaotic - Sheet1.tsv' 
lc_dataset = pd.read_csv(path, delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ge_corpus = []
for i in range(0, len(ge_dataset.index)):
    textinput = re.sub('[^a-zA-Z]', ' ', ge_dataset['Input'][i])
    textinput = textinput.lower()
    textinput = textinput.split()
    ps = PorterStemmer()
    all_stopwards = stopwords.words('english')
    all_stopwards.remove('not')
    textinput = [ps.stem(word) for word in textinput if not word in set(all_stopwards)]
    textinput = ' '.join(textinput)
    ge_corpus.append(textinput)

lc_corpus = []
for i in range(0, len(lc_dataset.index)):
    textinput = re.sub('[^a-zA-Z]', ' ', lc_dataset['Input'][i])
    textinput = textinput.lower()
    textinput = textinput.split()
    ps = PorterStemmer()
    all_stopwards = stopwords.words('english')
    all_stopwards.remove('not')
    textinput = [ps.stem(word) for word in textinput if not word in set(all_stopwards)]
    textinput = ' '.join(textinput)
    lc_corpus.append(textinput)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
ge_cv = CountVectorizer(max_features= 700)
ge_X = ge_cv.fit_transform(ge_corpus).toarray()
ge_y = ge_dataset.iloc[:, -1].values

lc_cv = CountVectorizer(max_features= 700)
lc_X = lc_cv.fit_transform(lc_corpus).toarray()
lc_y = lc_dataset.iloc[:, -1].values

# Training the Classifier Model on the Training Set
# SVM model
from sklearn.svm import SVC
ge_classifier = SVC(kernel='linear', random_state=0)
ge_classifier.fit(ge_X, ge_y)

# SVM model
from sklearn.svm import SVC
lc_classifier = SVC(kernel='linear', random_state=0)
lc_classifier.fit(lc_X, lc_y)

# Predicting the sentiments of a single sentence
new_sentence = input("Please enter a phrase (Enter Done to Finish): ")
while new_sentence != "Done":
    new_sentence = re.sub('[^a-zA-Z]', ' ', new_sentence)
    new_sentence = new_sentence.lower()
    new_sentence = new_sentence.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_sentence = [ps.stem(word) for word in new_sentence if not word in set(all_stopwords)]
    new_sentence = ' '.join(new_sentence)
    new_corpus = [new_sentence]
    new_ge_X_test = ge_cv.transform(new_corpus).toarray()
    new_ge_y_pred = ge_classifier.predict(new_ge_X_test)
    new_lc_X_test = lc_cv.transform(new_corpus).toarray()
    new_lc_y_pred = lc_classifier.predict(new_lc_X_test)
    if (new_lc_y_pred):
        print("Lawful")
    else:
        print("Chaotic")
    if (new_ge_y_pred):
        print("Good")
    else:
        print("Evil")
    new_sentence = input("Please enter a phrase (Enter Done to Finish): ")