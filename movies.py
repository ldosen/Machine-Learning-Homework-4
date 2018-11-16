import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn import metrics
from statistics import mean


class Vectorizer(object):

    def __init__(self, n=0, m=0):
        self.n = n
        self.m = m

    def extract_words(self, sentences, count_max, count_min):
        words_raw = []
        for sentence in sentences:
            words_raw.extend(sentence.split())

        # count all the words extracted
        word_counts = Counter(words_raw)

        # declare a final array that contains only the words we want to keep
        words = []
        for word in word_counts:
            count = word_counts[word]
            if count_max > count > count_min:
                words.append(word)

        return words

    def fit_transform(self, sentences):
        lexicon = self.extract_words(sentences, self.n, self.m)
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

    def __init__(self, model_params, vectorizer_params):
        self.model_params = model_params # a dict containing and array of parameters
        self.vectorizer_params = vectorizer_params # a dict containing and array of parameters

    def fit(self, X, y):

        all_scores = []
        k_fold = K_Fold()

        for i in range(len(self.vectorizer_params)):
            vect = Vectorizer(self.vectorizer_params.get('n')[i], self.vectorizer_params.get('m')[i])
            X_new = vect.fit_transform(X)
            for j in range(len(self.model_params)):
                model = LogisticRegression(C=self.model_params.get('C')[j])
                scores = k_fold.cross_validation(model, X_new, y, 10)
                all_scores.extend(mean(scores))

        sorted_all_scores = sorted(all_scores)
        return sorted_all_scores[0]


data = pd.read_csv('movie_reviews.csv')


# extract true class labels from features
X, y = data.iloc[:, 0].values, data.iloc[:, 1].values

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# transform the words in vectors
vect = Vectorizer(900, 10)
X_train = vect.fit_transform(X_train)
X_test = vect.fit_transform(X_test)

lreg = LogisticRegression(C=0.1)

lreg.fit(X_train, y_train)
score = lreg.score(X_train, y_train)


