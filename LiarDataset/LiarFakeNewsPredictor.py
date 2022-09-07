# This file uses the Liar dataset to predict fake news based on machine learning algorithms and natural
# language processing techniques
import pandas
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from FakeNewsFunctions import FakeNewsFunctions

# dataset used: LIAR Dataset
# dataset source: https://paperswithcode.com/dataset/liar
# dataset info source: https://paperswithcode.com/dataset/liar#:~:text=LIAR%20is%20a%20publicly%20available,fact%2Dchecking%20research%20as%20well.
# import the train and test data file from LIAR dataset
train_data = pandas.read_csv('train.tsv', sep='\t', header=None)
test_data = pandas.read_csv('test.tsv', sep='\t', header=None)

# add headings to all columns in the dataset for both testing and training datasets
train_data.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker job', 'state info',
                      'party affiliation',
                      'barely true count', 'false count', 'half true count', 'mostly true count', 'pants on fire count',
                      'context']

test_data.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker job', 'state info', 'party affiliation',
                     'barely true count', 'false count', 'half true count', 'mostly true count', 'pants on fire count',
                     'context']

# remove rows from dataset containing N/A values and reset the indexes of the new dataset
train_data.dropna(inplace=True)
train_data.reset_index(drop=True, inplace=True)

test_data.dropna(inplace=True)
test_data.reset_index(drop=True, inplace=True)

# split training and testing data into X(input) and y(output) values
# X represents the content of the article, y represents the  possible outcomes ranging from pants-fire, false, barelytrue, half-true, mostly-true, to true
X_train_data, y_train_data = train_data.iloc[:, 2], train_data.iloc[:, 1]
X_test_data, y_test_data = test_data.iloc[:, 2], test_data.iloc[:, 1]

# initialize class containing functions
fakenews_func = FakeNewsFunctions()

# convert text values to binary
y_train_binary = fakenews_func.convert_labels_to_binary(y_train_data)
y_test_binary = fakenews_func.convert_labels_to_binary(y_test_data)

# lemmatize X, the statement or content body of article
X_train_clean_data = fakenews_func.lemmatize_text(X_train_data)
X_test_clean_data = fakenews_func.lemmatize_text(X_test_data)

# initializing the Vectorizer
# source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# fit and transform the train and transform the test
X_tfidf_train = vectorizer.fit_transform(X_train_clean_data)
X_tfidf_test = vectorizer.transform(X_test_clean_data)

# initialize Logistic Regression model
# source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
lr_model = LogisticRegression()
lr_model.fit(X_tfidf_train, y_train_binary)  # train the LR model using training set
y_pred_lr = lr_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# initialize Decision Tree model
# source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
dt_model = DecisionTreeClassifier()
dt_model.fit(X_tfidf_train, y_train_binary)  # train the DT model using training set
y_pred_dt = dt_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# initialize SVM model
# source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
svm_model = svm.SVC()
svm_model.fit(X_tfidf_train, y_train_binary)  # train the SVM model using training set
y_pred_svm = svm_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

# Combining the 3 models to build a Voting Classifier using majority voting source:
# https://www.geeksforgeeks.org/ensemble-methods-in-python/#:~:text=Ensemble%20means%20a%20group%20of,robustness%2Fgeneralizability%20of%20the%20model.
ensemble_model = VotingClassifier(
    estimators=[('lr', lr_model), ('dt', dt_model), ('svm', svm_model)], voting='hard')

ensemble_model.fit(X_tfidf_train, y_train_binary)  # train the ensemble model using the training set
y_pred_ensemble = ensemble_model.predict(X_tfidf_test)  # input the X values from the test data to predict y-values

print('Liar Dataset Performance Metrics')
# source: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
print("LR Accuracy Score -> ",
      accuracy_score(y_pred_lr, y_test_binary) * 100)  # accuracy measured using sklearn metrics in % form
print("LR Precision Score -> ",
      precision_score(y_pred_lr, y_test_binary) * 100)  # precision measured using sklearn metrics in % form
print("LR Recall Score -> ",
      recall_score(y_pred_lr, y_test_binary) * 100)  # recall measured using sklearn metrics in % form
print("LR F1 Score -> ", f1_score(y_pred_lr, y_test_binary) * 100)  # f1 score measured using sklearn metrics in % form

print("DT Accuracy Score -> ", accuracy_score(y_pred_dt, y_test_binary) * 100)
print("DT Precision Score -> ", precision_score(y_pred_dt, y_test_binary) * 100)
print("DT Recall Score -> ", recall_score(y_pred_dt, y_test_binary) * 100)
print("DT F1 Score -> ", f1_score(y_pred_dt, y_test_binary) * 100)

print("SVM Accuracy Score -> ", accuracy_score(y_pred_svm, y_test_binary) * 100)
print("SVM Precision Score -> ", precision_score(y_pred_svm, y_test_binary) * 100)
print("SVM Recall Score -> ", recall_score(y_pred_svm, y_test_binary) * 100)
print("SVM F1 Score -> ", f1_score(y_pred_svm, y_test_binary) * 100)

print("Voted Classifier Accuracy Score -> ", accuracy_score(y_pred_ensemble, y_test_binary) * 100)
print("Voted Classifier Precision Score -> ", precision_score(y_pred_ensemble, y_test_binary) * 100)
print("Voted Classifier Recall Score -> ", recall_score(y_pred_ensemble, y_test_binary) * 100)
print("Voted Classifier F1 Score -> ", f1_score(y_pred_ensemble, y_test_binary) * 100)
