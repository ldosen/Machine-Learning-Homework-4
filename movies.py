import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.metrics import f1_score
from statistics import mean
import operator


class Vectorizer(object):

    def __init__(self, m=0):
        self.m = m

    def extract_words(self, sentences, count_min):
        words_raw = []
        for sentence in sentences:
            words_raw.extend(sentence.split())

        # count all the words extracted
        word_counts = Counter(words_raw)

        # declare a final array that contains only the words we want to keep
        words = []
        for word in word_counts:
            count = word_counts[word]
            if 900 > count > count_min:
                words.append(word)

        return words

    def fit_transform(self, sentences):
        lexicon = self.extract_words(sentences, self.m)
        featureset = []

        for w in sentences:
            words = w.split()
            features = np.zeros(len(lexicon))
            for i in words:
                if i in lexicon:
                    index_value = lexicon.index(i)
                    features[index_value] += 1

            featureset.append(features)

        featureset = np.array(featureset)
        return featureset


class K_Fold(object):

    def cross_validation(self, model, X, y, k):
        X_split = np.array_split(X, k)
        y_split = np.array_split(y, k)

        X_test = X_split[0]
        del X_split[0]

        y_test = y_split[0]
        del y_split[0]

        scores = []

        for fold in range(k - 1):
            model.fit(X_split[fold], y_split[fold])
            score = model.score(X_test, y_test)
            scores.append(score)

        return scores


class gridSearchTwo(object):

    def __init__(self, model, vectorizer, param_grid):
        self.model = model
        self.vectorizer = vectorizer
        self.param_grid = param_grid

    def fit(self, X, y):

        k_fold = K_Fold()
        score_params = {}

        for j in range(len(self.param_grid.get('m'))):
            vect = self.vectorizer(self.param_grid.get('m')[j])
            m = str(self.param_grid.get('m')[j])

            for i in range(len(self.param_grid.get('C'))):
                lreg = self.model(C=self.param_grid.get('C')[i])
                c = str(self.param_grid.get('C')[i])
                scores = k_fold.cross_validation(lreg, vect.fit_transform(X), y, 10)
                best_score = mean(scores)
                mc = m + ", " + c
                score_params.update({mc: best_score})

        return score_params


data = pd.read_csv('movie_reviews.csv')


# extract true class labels from features
X, y = data.iloc[:, 0].values, data.iloc[:, 1].values

vect = Vectorizer(10)
X = vect.fit_transform(X)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

'''
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
              'm': [10, 15, 20, 25, 30, 35, 40]}

gs = gridSearchTwo(LogisticRegression, Vectorizer, param_grid)

scores = gs.fit(X_train, y_train)
'''

lreg = LogisticRegression(C=1.0)
lreg.fit(X_train, y_train)
score = lreg.predict(X_test)
print("F1 Score:\n", (f1_score(y_test, score, average='macro')) * 100)
