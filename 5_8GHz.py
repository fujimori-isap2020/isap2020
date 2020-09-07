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

class Antenna(object):
    def __init__(self, x, y, freq, power, orientation, phase_offset):
        self.x_axis = x
        self.y_axis = y
        self.freq = freq
        self.power = power
        self.orientation = orientation
        c = 3.0e8
        self.lambda_0 = c / freq
        self.phase_offset = phase_offset

        # todo
        self.gain = 0

    def get_gain_by_angle(self, ant):
        return 1

    def get_distance_from(self, ant):
        x_diff = ant.x_axis - self.x_axis
        y_diff = ant.y_axis - self.y_axis
        return np.sqrt(x_diff**2 + y_diff**2)

    def get_distance_by_points(self, points):
        x_diff = points[:, 0] - self.x_axis
        y_diff = points[:, 1] - self.y_axis
        return np.sqrt(x_diff**2 + y_diff**2)

    def get_phase_from_distance(self, distance):
        return (distance % self.lambda_0) / self.lambda_0 * (2 * np.pi) + self.phase_offset

    def get_att_by_points(self, points):
        target_angles = atan2(points.x - self.x_axis, points.y - self.y_axis)
        pass

    def get_angles_by_points(self, points):
        pass

    def get_amplitude_by_points(self, points):
        pass

    def get_phase_by_ants(self, ants):
        distances = self.get_distance_from(ants)
        phases = self.get_phase_from_distance(distances)
        return phases


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
    dbuv = 20 * np.log10(voltage) + 120
    dbm = dbuv - 10 * np.log10(impedance) - 90
    return dbm

def calc_two_antennas(x1, y1, x2, y2, phase_diff):
    power_tx = 10
    gain_tx = 0
    # 受信側はダイポールで設定？？
    ant1 = Antenna(x1, y1, 5.8e9, 10, 90, 0)
    ant2 = Antenna(x2, y2, 5.8e9, 10, 90, phase_diff)
    tx_ants = [ant1, ant2]

    gain_rx = 0
    rx_ants = np.array([[0, -1], [0.5, -1], [1, -1], [1.5, -1], [2, -1], [2.5, -1], [3, -1]])

    complex_signals = list()
    for ant in tx_ants:
        # 常に各変数は受信アンテナ分のデータが入った配列になっている
        # 距離の取得
        distances = ant.get_distance_by_points(rx_ants)
        # 距離を使って受信点での位相の計算．
        phases = ant.get_phase_from_distance(distances)
        #print(max(phases/(2*np.pi)*360), min(phases/(2*np.pi)*360))
        # フリスの公式を使って受信点でのパワーを計算．
        power_rxs = received_power_by_friis(ant.power, ant.gain, gain_rx, distances, ant.lambda_0)
        # 振幅に変換
        amp_rxs = dbm_to_v(power_rxs, 50)
        # 振幅と位相から複素信号を計算
        complex_signal = np.sqrt(amp_rxs) * np.exp(1j * phases)
        complex_signals.append(complex_signal)

    # 受信アンテナごとに重ね合わせ
    sum_signals = np.sum(complex_signals, 0)
    return sum_signals


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


if __name__ == '__main__':
    deg_start = sys.argv[1]
    deg_stop = sys.argv[2]
    print(f'processing {deg_start} from {deg_stop}')
    for deg in tqdm(range(int(deg_start), int(deg_stop), 1)):
        rxpower = list()
        for x1 in tqdm(range(0, 300, 10)):
            for y1 in range(0, 300, 10):
                for x2 in range(0, 300, 10):
                    for y2 in range(0, 300, 10):
                        rxpower.append(calc_two_antennas(x1/100, y1/100, x2/100, y2/100, deg/360*np.pi*2))
        result = np.asarray(rxpower)
        with h5py.File(f'deg{deg}.hdf5', mode='w') as f:
            f.create_dataset(name='rxpowers', data=result)


    #print(test)
    #print(abs(np.array(test)))
    #plotter(test)




