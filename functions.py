import os
import numpy as np
import sklearn
import pandas as pd
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import random

# This class reads the data and splits it
class Data:
    def __init__(self, bodies, stances, alpha):
        self._bodies = bodies
        self._stances = stances
        self._alpha = alpha

    def process(self, csvs):
        array = []
        with open(csvs) as table:
            reader = csv.DictReader(table)
            for row in reader:
                array.append(row)

        return array

    def sort_data(self, bodies, stances):
        bods = {}
        for body in bodies:
            bods[int(body['Body ID'])] = body['articleBody']
        train_instances = []
        for i in range(len(stances)):
            a = [stances[i]["Headline"], bods[int(stances[i]["Body ID"])], stances[i]["Stance"]]
            train_instances.append(a)
        return train_instances

    def split_data(self):
        alpha = self._alpha
        body = self.process(self._bodies)
        stance = self.process(self._stances)
        data = self.sort_data(body, stance)
        random.shuffle(data)
        train_data = data[:int((1 - alpha) * len(data))]
        valid_data = data[int((1 - alpha) * len(data)):]
        return train_data, valid_data


# This function builds the vocabulary of the dataset
def build_vocab(a, b):
    corpus = a+b
    freq = defaultdict(float)
    for doc in corpus:
        unique = set(doc)
        for unique_word in unique:
            freq[unique_word] +=1.0

    filtered_words = {k:v for (k,v) in freq.items() if (v/len(corpus)) > 0.01 and (v/len(corpus))<0.75}
    vocab = set(filtered_words)
    return vocab

# TFIDF Vectorizer
class TFIDF:
    def __init__(self, dataset, vocab):
        self._dataset = dataset
        self._vocab = vocab
        self._word_index = {w: idx for idx, w in enumerate(self._vocab)}
        self._dataset_size = len(dataset)
        self._idf = np.zeros(len(vocab))
    
    def compute_tf_idf(self):
        for doc in self._dataset:
            indexes = [self._word_index[word] for word in set(doc) & self._vocab]
            self._idf[indexes] += 1.0
        self._idf = np.log(self._dataset_size / (1 + self._idf).astype(float))
        self._dataset = self._dataset.apply(lambda x: [self.tf_idf(x)])
        return self._dataset
    
    def tf_idf(self, document):
        tf_vector = np.zeros(len(self._vocab))
        for word in set(document) & self._vocab:
            tf = float(document.count(word)) / len(document)
            idf = self._idf[[self._word_index[word]]]
            if word not in self._word_index:
                tf_idf = 0
            else:
                tf_idf = tf * idf
            tf_vector[[self._word_index[word]]] = tf_idf
        return tf_vector    
    
    
# This function implements a laplace Unigram    
def Ngram(word, document, vocab):
    return (document.count(word) + 1)/ (len(document) + len(vocab))
    
    
# This function finds the common Ngrams in a headline/body pair    
def common_ngrams(body, headline, count):
    headline_ngrams = []
    article_ngrams = []

    for i in range(len(headline) - count):
        tup = headline[i:i+count]
        headline_ngrams.append(tuple(tup))

    for j in range(len(body) - count):
        tup = body[j:j+count]
        article_ngrams.append(tuple(tup))

    headline_ngrams = set(headline_ngrams)
    article_ngrams = set(article_ngrams)
    return len(article_ngrams & headline_ngrams)


# KL-Divergence
def KLD(vector_1, vector_2):
    x = [-vector_1[i] * np.log(vector_2[i]) for i in range(len(vector_1))]
    return np.sum(x)


# Cosine Similarity
def get_cosine(vec1, vec2):
    vec1 = np.asarray(vec1, dtype=np.float)
    vec2 = np.asarray(vec2, dtype=np.float)
    return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Jaccard Distance
def jaccard_distance(vec1, vec2):
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    binarize1 = np.where(vec1 > 0, 1, 0)
    binarize2 = np.where(vec2 > 0, 1, 0)
    added = binarize1 + binarize2
    intersection = np.sum(np.where(added >= 2, 1, 0))
    union = np.sum(added) - intersection
    return np.float(intersection) / (union + 0.01)


# Pearsons Correlation Co-efficient
def correlate(a,b):
    return np.corrcoef(a,b)[1,0]


# Manhattan Distance
def manhattan(a, b):
    return np.sum(np.absolute(np.array(a) - np.array(b)))


# Regression Function for both linear and logistic
class Regression:
    def __init__(self, features, regression="linear", learning_rate=.001, max_iterations=1000):
        self._size = features
        self._learning_rate = learning_rate
        self._regression = regression
        self._iterations = max_iterations
        self._weights = np.zeros((self._size, 1))

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def predict(self, X):
        if self._regression == "logistic":
            y_pred = self.sigmoid(np.dot(X, self._weights))
        elif self._regression == "linear":
            y_pred = np.dot(X, self._weights)
        return y_pred

    def calculate_loss(self, X, Y):
        return np.sum((self.predict(X) - Y) ** 2) / (2 * np.shape(X)[0])

    def gradient_descent(self, X, Y):
        error = self.predict(X) - Y
        gradient = np.dot(X.T, error) / np.shape(X)[0]
        self._weights = self._weights - self._learning_rate * gradient

    def train_model(self, X, Y):
        old_loss = 0
        for i in range(self._iterations):
            self.gradient_descent(X, Y)
            loss = self.calculate_loss(X, Y)
            print('Iteration: ' + str(i))
            print('Loss: ' + str(loss))
            if np.abs(loss - old_loss) < 0.001 * loss:
                print("Early Stopped at Epoch: ", i)
                print("Final Loss:", loss)
                break
            loss = old_loss
            
        
# F1-score         
def f_score(data, predictions):
    confusion = defaultdict(int)
    for y_gold, y_guess in zip(data, predictions):
        confusion[(y_gold, y_guess)] += 1
        
    tp, fp, tn = 0,0,0    
    
    for (gold, guess), count in confusion.items():
        if guess != gold:
            fp += count
            tn += count
        elif guess == gold:
            tp += count
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) 
    return f1