import numpy as np
import pandas as pd

result = (np.load('result.npy')).reshape(7,810000).T
positions = (np.load('positions.npy')).reshape(4,810000).T

dataset = np.concatenate([result, positions], 1)
np.random.shuffle(dataset)

train, test = np.split(dataset, [int(810000 * 0.8)])

train = pd.DataFrame(train)
test = pd.DataFrame(test)

train.columns = ['x__0', 'x__1', 'x__2', 'x__3', 'x__4', 'x__5', 'x__6', 'y__0', 'y__1', 'y__2', 'y__3']
test.columns = ['x__0', 'x__1', 'x__2', 'x__3', 'x__4', 'x__5', 'x__6', 'y__0', 'y__1', 'y__2', 'y__3']

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
