import nltk
from nltk.corpus import stopwords
from nltk.stem.cistem import Cistem
import re
import string


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
    stops = stopwords.words('german') + list(string.punctuation)
    return [i for i in text if i not in stops]


def stem(text):
    stemmer = Cistem()
    for index, word in enumerate(text):
        text[index] = stemmer.stem(word)
    return text


def preprocessing(text):
    text = remove_numbers(text)
    text = lower_case(text)
    if type(text) is str:
        text = tokenize(text)
    text = remove_stop_words(text)
    text = stem(text)
    return text
