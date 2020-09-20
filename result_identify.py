import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py


#xの範囲．10cm刻み．yも同じ
d = np.arange(0, 300, 10)

#xとyそれぞれの距離の差を7x30の行列にいれる
x = d.reshape(1,30)
y = d.reshape(1,30)

#xとyで向きが違う分を，新しい要素を追加するときに考慮
x = x[:, :, np.newaxis]
y = y[:, np.newaxis, :]

base = np.zeros([30, 30])
x1 = (base + x).reshape(1,900)
y1 = (base + y).reshape(1,900)
x2 = x1
y2 = y1

result = np.zeros([4,900,900])

for i in range(900):
    result[0,:,i] = x1
    result[1,:,i] = y1

for i in range(900):
    result[2,i,:] = x2
    result[3,i,:] = y2
    
#with h5py.File(f'positions.hdf5', mode='w') as f:
#    f.create_dataset(name='position', data=result)  #resultがcomplex型なので予想の2倍？

np.save('positions.npy', result)