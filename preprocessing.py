import nltk
from nltk.corpus import stopwords
from nltk.stem.cistem import Cistem
import re
import string
import pickle
import numpy as np


def remove_numbers(text):
    if type(text) is list:
        for index, word in enumerate(text):
            text[index] = re.sub(r'\d+', '', word)
    else:
        text = re.sub(r'\d+', '', text)
    return text


def lower_case(text):
    if type(text) is list:
        for index, word in enumerate(text):
            text[index] = word.lower()
    else:
        text = text.lower()
    return text


def tokenize(text):
    return nltk.tokenize.WordPunctTokenizer().tokenize(text)


def remove_stop_words(text):
    stops = stopwords.words('german') + stopwords.words('english') + list(string.punctuation)
    return [i for i in text if i not in stops]


def stem(text):
    stemmer = Cistem()
    for index, word in enumerate(text):
        text[index] = stemmer.stem(word)
    return text


def preprocessing_text(text):
    text = remove_numbers(text)
    text = lower_case(text)
    if type(text) is str:
        text = tokenize(text)
    text = remove_stop_words(text)
    text = stem(text)
    return text


def preprocessing(text):
    preprocessed_text = preprocessing_text(text)
    with open('data/123456789/keywords.dictionary', 'rb') as keywords_dictionary:
        keywords = pickle.load(keywords_dictionary)
    with open('data/123456789/adjecency_matrix.dictionary', 'rb') as adjecency_matrix_dictionary:
        adjacency_matrix = pickle.load(adjecency_matrix_dictionary)
    with open('data/123456789/categories.dictionary', 'rb') as categories_dictionary:
        categories = pickle.load(categories_dictionary)
    feature = np.zeros(len(categories))
    for word in preprocessed_text:
        try:
            index = keywords.index(word)
            feature = feature + adjacency_matrix[index]
        except ValueError:
            continue
    feature_length = np.linalg.norm(feature)
    if feature_length > 0:
        feature = feature / feature_length
    return feature


def get_index_of_category(category):
    with open('data/123456789/categories.dictionary', 'rb') as categories_dictionary:
        categories = pickle.load(categories_dictionary)
    return categories.index(category)