from sklearn.naive_bayes import MultinomialNB
import numpy as np
import csv
from preprocessing import preprocessing, get_index_of_category

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])

def read_train_data():
    print('Load training data:')
    with open('data/123456789/train-data.csv', 'r') as train_csv:
        reader = csv.reader(train_csv, delimiter=';')
        for index, row in enumerate(reader):
            if index == 0:
                X_train = np.array([preprocessing(row[0])])
                Y_train = np.array([get_index_of_category(row[1])])
            else:
                X_train = np.vstack((X_train, preprocessing(row[0])))
                Y_train = np.vstack((Y_train, get_index_of_category(row[1])))
    return X_train, Y_train


def read_test_data():
    print('Load testing data:')
    with open('data/123456789/test-data.csv', 'r') as test_csv:
        reader = csv.reader(test_csv, delimiter=';')
        for index, row in enumerate(reader):
            if index == 0:
                X_test = np.array([preprocessing(row[0])])
                Y_test = np.array([get_index_of_category(row[1])])
            else:
                X_test = np.vstack((X_test, preprocessing(row[0])))
                Y_test = np.vstack((Y_test, get_index_of_category(row[1])))
    return X_test, Y_test


def train():
    X_train, Y_train = read_train_data()
    X_test, Y_test = read_test_data()
    print('Start training')
    print(X_train.shape)
    clf = MultinomialNB().fit(X_train, Y_train.ravel())
    predicted = clf.predict(X_test)
    accuracy = np.mean(predicted == Y_test)
    return accuracy


print('Accuracy: ', train())