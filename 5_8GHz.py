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

import configparser

# self-made function
from propagation import raytrace as rt
import const


class Antennas(object):

    def __init__(self, x, y, z, freq, power, orientation, phase_offset):
        self.x_position = x         # array of 30 : range(0, 300, 10)  [cm]
        self.y_position = y         # array of 30 : range(0, 300, 10)  [cm]
        self.y_position = z         # array of 30 : range(0, 300, 10)  [cm]
        self.freq = freq
        self.power = power
        self.orientation = orientation
        self.lambda_0 = const.c / freq
        self.phase_offset = phase_offset

        # todo
        self.gain = const.tx_gain


def calc_two_antennas_vector(phase_diff):
    x1_grid = np.linspace(const.x_start, const.x_range, const.x_num)
    y1_grid = np.linspace(const.y_start, const.y_range, const.y_num)
    x2_grid = np.linspace(const.x_start, const.x_range, const.x_num)
    y2_grid = np.linspace(const.y_start, const.y_range, const.y_num)

    # 受信側はダイポールで設定？？
    # Antennasはx,yのgridに対してとりうる全ての点を保持する．
    ant1 = Antennas(x1_grid, y1_grid, const.freq,
                    const.tx_power, const.orientation, const.phase_of_origin)
    ant2 = Antennas(x2_grid, y2_grid, const.freq, const.tx_power,
                    const.orientation, phase_diff)
    tx_ants = [ant1, ant2]

    gain_rx = const.rx_gain

    # 受信アンテナの位置
    # Todo : Antennasクラスを使って作りたいが，get_distance_by_pointsと密接につながっているため，そこと同時に編集する必要がある．
    # x軸の位置
    rx_ants_x_position = np.linspace(
        const.point_of_origin, const.x_range, const.rx_antenna_num)
    # y軸の位置
    rx_ants_y_position = np.full(
        const.rx_antenna_num, const.rx_ants_y_position)
    rx_ants_position = np.stack([rx_ants_x_position, rx_ants_y_position], 1)

    complex_signals = list()

    for ant in tx_ants:
        # ７個の各アンテナからantのとりうる位置30x30点に対する距離の取得 (distances.shape = (900, 7))
        distances = rt.get_distance_by_points(
            rx_ants_position, tx_ants, const.x_num, const.y_num)
        # 距離を使って受信点での位相の計算．(phases.shape = (900, 7))
        phases = rt.get_phase_from_distance(distances, const.lambda_0, tx_ants)
        #print(max(phases/(2*np.pi)*360), min(phases/(2*np.pi)*360))
        # 2波モデルを使って受信点でのパワーを計算．
        # 距離を使って受信点での位相の計算．(power_rxs.shape = (900, 7))
        power_rxs = rt.receive_voltage_gain(
            distances, const.lambda_0, const.k, const.tx_antenna_hight, const.rx_antenna_hight)
        # 振幅に変換 (power_rxs.shape = (900, 7))
        amp_rxs = rt.dbm_to_v(power_rxs, 50)
        # 振幅と位相から複素信号を計算 (power_rxs.shape= (900, 7))
        complex_signal = np.sqrt(amp_rxs) * np.exp(1j * phases)
        complex_signals.append(complex_signal)

    # complex_signals.shape -> (2, 900, 7)
    complex_signals = np.array(complex_signals)
    # 受信アンテナごとに重ね合わせ (complex_signals[0] ? complex_signals[1]).shape -> (810000, 7)
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
        plt.plot([0, 0.5, 1, 1.5, 2, 2.5, 3], rt.v_to_dbm(
            abs(test[i]), 50), label=f'ant1.y={i*0.1}')
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
        calc_two_antennas_vector(deg)
