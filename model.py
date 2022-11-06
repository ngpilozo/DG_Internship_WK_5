import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('price.csv')
X = dataset.iloc[:, :3]
X['bed_room'].fillna(0, inplace=True)
X['area'].fillna(dataset['area'].mean(), inplace=True)

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['bed_room'] = X['bed_room'].apply(lambda x : convert_to_int(x))

"""
def find_next_lowest(val, lis):
    lis.sort()
    next_lowest = val
    for x in reversed(lis):
        if x < next_lowest:
            next_lowest = x
            return next_lowest

def find_same(val, bed_rooms, lis):
    if not(val.isnull()):
        return val

    x = lis[~lis.loc['{}'.format(lis['area'])].isnull()]
    if len(x) < 1:
        return lis[~lis.loc['{}'.format(find_next_lowest(lis['area'], lis))].isnull()]
    else:
        return lis[1].mean()

X['area'] = X[['area', 'bed_room']].apply(lambda x: find_same(*x, X), axis=1)
"""
Y = dataset.iloc[:, -1]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

print(X)
print(Y)
regressor.fit(X.values, Y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 2200, 5]]))


