import numpy as np
import pandas as pd

result = (np.load('result.npy')).reshape(7,810000).T
positions = (np.load('positions.npy')).reshape(4,810000).T

dataset = np.concatenate([result, positions], 1)
#dataset = dataset[:3]
dataset = pd.DataFrame(dataset)
dataset.columns = ['x__0', 'x__1', 'x__2', 'x__3', 'x__4', 'x__5', 'x__6', 'y__0', 'y__1', 'y__2', 'y__3']
print(dataset.info())

dataset.to_csv('dataset.csv', index=False)
#np.savetxt('dataset.csv', dataset, delimiter=',')