from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import csv
import numpy as np
from preprocessing import get_index_of_category
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.stem.cistem import Cistem
from nltk.corpus import stopwords
import pickle

def tokenize(text):
    return nltk.tokenize.WordPunctTokenizer().tokenize(text)

def stem(text):
    if type(text) is str:
        text = tokenize(text)
    stemmer = Cistem()
    for index, word in enumerate(text):
        text[index] = stemmer.stem(word)
    return ' '.join(text)

def read_train_data():
    print('Load training data:')
    with open('data/123456789/train-data.csv', 'r') as train_csv:
        reader = csv.reader(train_csv, delimiter=';')
        X_train = []
        Y_train = []
        for index, row in enumerate(reader):
            X_train.append(stem(row[0]))
            Y_train.append(get_index_of_category(row[1]))
    return X_train, Y_train

def read_test_data():
    print('Load testing data:')
    with open('data/123456789/test-data.csv', 'r') as train_csv:
        reader = csv.reader(train_csv, delimiter=';')
        X_test = []
        Y_test = []
        for index, row in enumerate(reader):
            X_test.append(stem(row[0]))
            Y_test.append(get_index_of_category(row[1]))
    return X_test, Y_test

# count_vect = CountVectorizer()
X_train, Y_train = read_train_data()
X_test, Y_test = read_test_data()
# text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('german'))), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier())])
# text_clf = text_clf.fit(X_train, Y_train)
# predicted = text_clf.predict(X_test)
# accuracy = np.mean(predicted == Y_test)
#
# print('Accuracy: ', accuracy)

# with open('data/123456789/model.dictionary', 'wb') as model_dictionary:
#     pickle.dump(text_clf, model_dictionary)

with open('data/123456789/model.dictionary', 'rb') as model_dictionary:
    text_clf = pickle.load(model_dictionary)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == Y_test)

    print('Accuracy: ', accuracy)


############

# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3)}
#
# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(X_train, Y_train)
# print(gs_clf.best_score_)
# print(gs_clf.best_params_)