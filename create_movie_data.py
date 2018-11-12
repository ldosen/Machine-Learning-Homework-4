import numpy as np
import pandas as pd


# coerce the data into a pandas dataframe and save it as a .csv file

df = pd.DataFrame()

with open('rt-polarity.pos', 'r') as f:
    contents = f.readlines()
    for i in contents:
        df = df.append([[i, 1]], ignore_index=True)

with open('rt-polarity.neg', 'r') as f:
    contents = f.readlines()
    for i in contents:
        df = df.append([[i, 0]], ignore_index=True)

df.columns = ['review', 'sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_reviews.csv', index=False)


