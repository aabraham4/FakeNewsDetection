# This file uses the Kaggle FakeNews dataset to predict fake news based on machine learning algorithms and natural
# language processing techniques
import pandas
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from FakeNewsFunctions import FakeNewsFunctions

# dataset used: Fake News dataset from kaggle
# source of dataset: https://www.kaggle.com/c/fake-news
# import fake news dataframe from Kaggle Fake News dataset
fakenews_df = pandas.read_csv('train.csv')

# remove any rows with N/A values and update dataset index
fakenews_df.dropna(inplace=True)
fakenews_df.reset_index(drop=True, inplace=True)

# select columns corresponding to text as X and target value as y
# X represents the text body of each article
X = fakenews_df.iloc[:, 3]
# y is a binary representation of output label, 1 meaning the article is unreliable, 0 meaning the article is reliable
y = fakenews_df.iloc[:, 4]

# lemmatize the text body of X
fakenews_func = FakeNewsFunctions()
lemmatized_X = fakenews_func.lemmatize_text(X)

# split X and y data into train and test data with 0.8:0.2 train to test ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# initializing the Vectorizer
# source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# fit and transform the train and transform the test
X_tfidf_train = vectorizer.fit_transform(X_train)
X_tfidf_test = vectorizer.transform(X_test)

# initialize Logistic Regression model
# source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
lr_model = LogisticRegression()
lr_model.fit(X_tfidf_train, y_train)  # train the LR model using training set
y_pred_lr = lr_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# initialize Decision Tree model
# source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
dt_model = DecisionTreeClassifier()
dt_model.fit(X_tfidf_train, y_train)  # train the DT model using training set
y_pred_dt = dt_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# initialize SVM model
# source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
svm_model = svm.SVC()
svm_model.fit(X_tfidf_train, y_train)  # train the SVM model using training set
y_pred_svm = svm_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# Combining the 3 models to build a Voting Classifier using majority voting source:
# https://www.geeksforgeeks.org/ensemble-methods-in-python/#:~:text=Ensemble%20means%20a%20group%20of,robustness%2Fgeneralizability%20of%20the%20model.
ensemble_model = VotingClassifier(
    estimators=[('lr', lr_model), ('dt', dt_model), ('svm', svm_model)], voting='hard')

ensemble_model.fit(X_tfidf_train, y_train)  # train the ensemble model using the training set
y_pred_ensemble = ensemble_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# code to display text, actual vs predicted value
actual_vs_pred_df = pandas.DataFrame()
actual_vs_pred_df['text'] = X_test
actual_vs_pred_df['actual label'] = y_test
actual_vs_pred_df['predicted label'] = y_pred_svm
pandas.set_option('display.max_columns', None)  # displays all columns without leaving out any
# print(actual_vs_pred_df.head())

print('Kaggle Fake News Dataset Performance Metrics')

# source: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
print("LR Accuracy Score -> ",
      accuracy_score(y_pred_lr, y_test) * 100)  # accuracy measured using sklearn metrics in % form
print("LR Precision Score -> ",
      precision_score(y_pred_lr, y_test) * 100)  # precision measured using sklearn metrics in % form
print("LR Recall Score -> ", recall_score(y_pred_lr, y_test) * 100)  # recall measured using sklearn metrics in % form
print("LR F1 Score -> ", f1_score(y_pred_lr, y_test) * 100)  # f1 score measured using sklearn metrics in % form

print("DT Accuracy Score -> ", accuracy_score(y_pred_dt, y_test) * 100)
print("DT Precision Score -> ", precision_score(y_pred_dt, y_test) * 100)
print("DT Recall Score -> ", recall_score(y_pred_dt, y_test) * 100)
print("DT F1 Score -> ", f1_score(y_pred_dt, y_test) * 100)

print("SVM Accuracy Score -> ", accuracy_score(y_pred_svm, y_test) * 100)
print("SVM Precision Score -> ", precision_score(y_pred_svm, y_test) * 100)
print("SVM Recall Score -> ", recall_score(y_pred_svm, y_test) * 100)
print("SVM F1 Score -> ", f1_score(y_pred_svm, y_test) * 100)

print("Voted Classifier Accuracy Score -> ", accuracy_score(y_pred_ensemble, y_test) * 100)
print("Voted Classifier Precision Score -> ", precision_score(y_pred_ensemble, y_test) * 100)
print("Voted Classifier Recall Score -> ", recall_score(y_pred_ensemble, y_test) * 100)
print("Voted Classifier F1 Score -> ", f1_score(y_pred_ensemble, y_test) * 100)
