import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn import metrics
from random import randrange


class Vectorizer(object):

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

    def fit_transform(self, sentences, n, m):
        lexicon = self.extract_words(sentences, n, m)
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


data = pd.read_csv('movie_reviews.csv')


# extract true class labels from features
X, y = data.iloc[:, 0].values, data.iloc[:, 1].values

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# transform the words in vectors
vect = Vectorizer()
X_train = vect.fit_transform(X_train, 20, 10)
X_test = vect.fit_transform(X_test, 20, 10)

lreg = LogisticRegression()

kfold = K_Fold()

scores = kfold.cross_validation(lreg, X_train, y_train, 10)
print("k-fold scores:\n", scores)

