import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn import metrics


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


class Cross_Validation(object):

    def partition(self, vector, fold, k):
        size = int(vector.shape[0])
        start = int((size/k)*fold)
        end = int((size/k)*(fold+1))
        validation = vector[start:end]

        if str(type(vector)) == "<class 'scipy.sparse.csr.csr_matrix'>":
            indices = range(start, end)
            mask = np.ones(vector.shape[0], dtype=bool)
            mask[indices] = False
            training = vector[mask]
        elif str(type(vector)) == "<type 'numpy.ndarray'>":
            training = np.concatenate((vector[:start], vector[end:]))
        return training, validation

    def cross_validation(self, learner, k, data, labels):
        train_folds_score = []
        validation_folds_score = []

        for fold in range(k):
            # create training, validation sets/labels
            training_set, validation_set = self.partition(data, fold, k)
            training_labels, validation_labels = self.partition(labels, fold, k)

            # fit the machine learning model to the new sets and predict
            learner.fit(training_set, training_labels)
            training_predicted = learner.predict(training_set)
            validation_predicted = learner.predict(validation_set)

            # add the scores to the record
            train_folds_score.append(metrics.accuracy_score(training_labels, training_predicted))
            validation_folds_score.append(metrics.accuracy_score(validation_labels, validation_predicted))

        return train_folds_score, validation_folds_score


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
cv = Cross_Validation()

score = cv.cross_validation(lreg, 10, X_train, y_train)
print("Train score", str(score[0]))
print("Test score", str(score[1]))
