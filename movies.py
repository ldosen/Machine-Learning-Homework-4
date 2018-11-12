import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter


def extract_words(sentences, count_max, count_min):
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


def fit_transform(sentences, n, m):
    lexicon = extract_words(sentences, n, m)
    features = np.zeros(len(lexicon))

    for w in sentences:
        words = w.split()
        for i in words:
            if i in lexicon:
                index_value = lexicon.index(i)
                features[index_value] += 1

    return features




data = pd.read_csv('movie_reviews.csv')
x = data.iloc[:, 0]

test = fit_transform(x, 20, 10)

print(test)

