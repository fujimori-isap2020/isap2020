import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py


#constants 定数
c = 3.0e8               #光速
freq = 5.8e9            #周波数
lambda_0 = c / freq     #波長λ
impedance = 50
degree = cp.arange(0, 360, 1)   #位相差用角度
amplitude = 1           #振幅
Tx_Power = 1
Gain_Tx = 0             #ゲイン
Gain_Rx = 0             #dBなので0は無増幅

#受信点
rx_ants = cp.array([[0, -1], [0.5, -1], [1, -1], [1.5, -1], [2, -1], [2.5, -1], [3, -1]])

#xの範囲．10cm刻み．yも同じ
d = cp.arange(0, 300, 10)

#xとyそれぞれの距離の差を7x30の行列にいれる
x_diff = rx_ants[:,0].reshape(7,1) - d.reshape(1,30)
y_diff = rx_ants[:,1].reshape(7,1) - d.reshape(1,30)

#xとyで向きが違う分を，新しい要素を追加するときに考慮
x_diff = x_diff[:, :, cp.newaxis]
y_diff = y_diff[:, cp.newaxis, :]

#距離をsqrtで求めて7x30x30を7x900に変形
distances = cp.sqrt(x_diff**2 + y_diff**2) / 100   #7x30x30  100はcmからmへの変換
phases_0 = (distances % lambda_0) / lambda_0 * 2 * cp.pi    #基準送信源

Gf = -10 * cp.log10((4*cp.pi*distances/lambda_0)**2) + (Gain_Tx + Gain_Rx)  # 自由空間のゲイン
Rx_P = Tx_Power + Gf
dbuV = Rx_P + 10 * cp.log10(impedance) + 90     #dbm_to_v関数の中身
Rx_Amp = (10 ** (dbuV/20)) / 1e6

complex_signal_0 = cp.sqrt(Rx_Amp) * cp.exp(1j * phases_0)  #基準送信源の複素信号
comp_0 = complex_signal_0.reshape(7, 900)   #7x900に変形            
result = cp.zeros([7,900,900], dtype=cp.complex)    #計算結果を入れる箱

#comp_0を基準の信号源としている．それを先に計算しておき，それに対して足し合わせる位相差ありの
#他の信号源はこれ以後のfor文のなかで計算する．

#一番外側のfor文
for i in tqdm(range(int(0), int(360), 1)):
    phase_offset = (i / 360 * cp.pi * 2)
    phases = phases_0 + phase_offset        #基準プラス位相差
    
    #この時点で距離と位相が用意できているので，次は電力の計算(位相差ありの方)
    complex_signal = cp.sqrt(Rx_Amp) * cp.exp(1j * phases)

    #全パターンの足し合わせ
    comp = complex_signal.reshape(7, 900)
    for j in range(900):
        result[:,:,j] = comp_0 + comp[:,j].reshape(7,1)
    
'''
with h5py.File(f'degree.hdf5', mode='w') as f:
    f.create_dataset(name='rxpowers', data=result)  #resultがcomplex型なので予想の2倍？
'''
