# this program is made for sdc in isap2020
# transmitter A: 300-320MHz, -50dBm, lambda/2 dipole, horizontal, 1-3antennas, CW
# transmitter B: 2.402-2.480GHz, 0dBm, Small antenna, unknown pole, 1-3antennas, ble beacon
# transmitter C: 5.012-5.025GHz, 10dBm, Monopole antenna, Vertical, 1-2, CW
#

import numpy as np
import pandas as pd
from math import atan2
import matplotlib.pyplot as plt

from tqdm import tqdm
import pickle
import sys
import h5py


class Antennas(object):

    def __init__(self, x, y, freq, power, orientation, phase_offset):
        # x should be array
        # y should be array

        self.x_axis = x         # array of 30 : range(0, 300, 10)  [m]
        self.y_axis = y         # array of 30 : range(0, 300, 10)  [m]
        self.freq = freq
        self.power = power
        self.orientation = orientation
        c = 3.0e8
        self.lambda_0 = c / freq
        self.phase_offset = phase_offset

        # todo
        self.gain = 0

    def get_gain_by_angle(self, ant):
        # returns (900)
        return 1

    def get_distance_by_points(self, points):
        # returns (900, 7)
        # xとyそれぞれの距離の差を7x30の行列にいれる by kobayashi
        x_diff = points[:, 0].reshape(7, 1) - self.x_axis.reshape(1, 30)
        y_diff = points[:, 1].reshape(7, 1) - self.y_axis.reshape(1, 30)
        # xとyで向きが違う分を，新しい要素を追加するときに考慮 by kobayashi
        x_diff = x_diff[:, :, np.newaxis]
        y_diff = y_diff[:, np.newaxis, :]
        # 単位はメートルで返す．
        distances = np.sqrt(x_diff**2 + y_diff**2) / 100
        return distances.reshape(7, 900)

    def get_phase_from_distance(self, distance):
        # returns (900, 7)
        return (distance % self.lambda_0) / self.lambda_0 * (2 * np.pi) + self.phase_offset


def received_power_by_friis(power_tx, gain_tx, gain_rx, distance, lambda_0):
    # 自由空間のゲイン
    gf = -10 * np.log10((4 * np.pi * distance / lambda_0) ** 2) + (gain_tx + gain_rx)
    received_power = power_tx + gf
    return received_power


def dbm_to_v(power_dbm, impedance):
    dbuV = power_dbm + 10 * np.log10(impedance) + 90
    voltage = (10 ** (dbuV/20)) / 1e6
    return voltage


def v_to_dbm(voltage, impedance):
    dbuV = 20 * np.log10(voltage) + 120
    dbm = dbuV - 10 * np.log10(impedance) - 90
    return dbm


def plotter(test):
    plt.figure(dpi=200, figsize=(5, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams["legend.markerscale"] = 2
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.edgecolor"] = 'black'
    for i in range(10):
        plt.plot([0, 0.5, 1, 1.5, 2, 2.5, 3], v_to_dbm(abs(test[i]), 50), label=f'ant1.y={i*0.1}')
    plt.title('ant1=(0, 0.0-0.1), ant2=(1.2, 1.5) [m]')
    plt.ylabel('rx power [dBm]')
    plt.xlabel('mesurement point x [m], (y=0 [m])')
    plt.grid()
    plt.legend()
    plt.show()


def calc_two_antennas_vector(phase_diff):
    x1_grid = np.arange(0, 300, 10)
    y1_grid = np.arange(0, 300, 10)
    x2_grid = np.arange(0, 300, 10)
    y2_grid = np.arange(0, 300, 10)

    # 受信側は等方性で仮定
    # Antennasはx,yのgridに対してとりうる全ての点を保持する．
    ant1 = Antennas(x1_grid, y1_grid, 5.8e9, 10, 90, 0)
    ant2 = Antennas(x2_grid, y2_grid, 5.8e9, 10, 90, phase_diff)
    tx_ants = [ant1, ant2]

    gain_rx = 0
    rx_ants = np.array([[0, -1], [0.5, -1], [1, -1], [1.5, -1], [2, -1], [2.5, -1], [3, -1]])

    complex_signals = list()
    for ant in tx_ants:
        # ７個の各アンテナからantのとりうる位置30x30点に対する距離の取得 (distances.shape = (900, 7))
        distances = ant.get_distance_by_points(rx_ants)
        print(f'distances.shape: {distances.shape}')
        # 距離を使って受信点での位相の計算．(phases.shape = (900, 7))
        phases = ant.get_phase_from_distance(distances)
        print(f'phases.shape: {phases.shape}')
        #print(max(phases/(2*np.pi)*360), min(phases/(2*np.pi)*360))
        # フリスの公式を使って受信点でのパワーを計算．
        # 距離を使って受信点での位相の計算．(power_rxs.shape = (900, 7))
        power_rxs = received_power_by_friis(ant.power, ant.gain, gain_rx, distances, ant.lambda_0)
        # 振幅に変換 (power_rxs.shape = (900, 7))
        amp_rxs = dbm_to_v(power_rxs, 50)
        # 振幅と位相から複素信号を計算 (power_rxs.shape= (900, 7))
        complex_signal = np.sqrt(amp_rxs) * np.exp(1j * phases)
        complex_signals.append(complex_signal)

    # complex_signals.shape -> (2, 7, 900)
    complex_signals = np.array(complex_signals)
    # 受信アンテナごとに重ね合わせ (complex_signals[0] ? complex_signals[1]).shape -> (810000, 7)
    sum_signals_for_each_rx = complex_signals[0][:, :, None] + complex_signals[1][:, None, :]
    return sum_signals_for_each_rx


if __name__ == '__main__':
    deg_start = sys.argv[1]
    deg_stop = sys.argv[2]
    print(f'processing {deg_start} from {deg_stop}')
    for deg in tqdm(range(int(deg_start), int(deg_stop), 1)):
        result = calc_two_antennas_vector(deg)
        with h5py.File(f'dataset_tmp/degree{deg}.hdf5', mode='w') as f:
            f.create_dataset(name='rxpowers', data=result)  # resultがcomplex型なので予想の2倍？
            # データが大きくなると予想されるので，forを出てから最後のデータだけを保存





