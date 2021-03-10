# Juan Davalos and Michael Williamson
# Computer Science 542 - Machine Learning

# Natural Language Processing
# Sentiment Analysis

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Importing the dataset
path = Path(__file__).parent / 'Good_Bad - Sheet1.tsv'
dataset = pd.read_csv(path, delimiter = '\t', quoting = 3)
#processes tsv files, and ignores double quotes (")

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset.index)):
    textinput = re.sub('[^a-zA-Z]', ' ', dataset['Input'][i])
    textinput = textinput.lower()
    textinput = textinput.split()
    ps = PorterStemmer()
    all_stopwards = stopwords.words('english')
    all_stopwards.remove('not')
    #textinput = [ps.stem(word) for word in textinput if not word in set(all_stopwards)]
    textinput = ' '.join(textinput)
    corpus.append(textinput)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Logistic Regression model on the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)

# Training the KNN model on the Training set
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier()
# classifier.fit(X_train, y_train)

# Training the SVM on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Training the Kernal SVM on the Training set
# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)

# Training the Naive Bayes model on the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# Training the Decision Tree on the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion="entropy", random_state= 0)
# classifier.fit(X_train, y_train)

# Training the Random Forest on the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state= 0)
# classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
cm = confusion_matrix(y_test, classifier.predict(X_test))
print(cm)
accuracy = accuracy_score(y_test, classifier.predict(X_test))
print(accuracy)
stats = precision_recall_fscore_support(y_test, classifier.predict(X_test), average='binary')
print(stats)