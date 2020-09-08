# Calculation of Electric Field Strength Using Ray Trace Method
# Powered by Fujimori-Lab Members (Akada, Kobayashi, Maruyama) in September 2020

import numpy as np


def received_power_by_friis(power_tx, gain_tx, gain_rx, distance, lambda_0):
    # 自由空間のゲイン
    gf = -10 * np.log10((4 * np.pi * distance / lambda_0)
                        ** 2) + (gain_tx + gain_rx)
    received_power = power_tx + gf
    return received_power


def dbm_to_v(power_dbm, impedance):
    # 電力電圧変換
    dbuV = power_dbm + 10 * np.log10(impedance) + 90
    voltage = (10 ** (dbuV/20)) / 1e6
    return voltage


def v_to_dbm(voltage, impedance):
    # 電圧電力変換
    dbuV = 20 * np.log10(voltage) + 120
    dbm = dbuV - 10 * np.log10(impedance) - 90
    return dbm


def path_length_difference(distance, ht, hr):
    # 行路長差
    delta_l = np.sqrt((ht + hr)**2 + distance**2) - \
        np.sqrt((ht - hr) ** 2 + distance ** 2)
    return delta_l


def receive_voltage_gain(distance, lambda_0, k, ht, hr):
    # 受信電圧利得
    Ge = 20 * np.log10(np.sqrt(lambda_0 / (4 * np.pi*distance)
                               * abs(1 - np.exp(1j * k * path_length_difference(distance, ht, hr)))))
    return Ge


def get_gain_by_angle(ant):
    # returns (900)
    return 1


def get_distance_from(rx_ants_position, tx_ants, x_num, y_num):
    # returns (900)
    # xとyそれぞれの距離の差を7x30の行列にいれる by kobayashi
    x_diff = rx_ants_position[:, 0].reshape(
        7, 1) - tx_ants.x_position.reshape(1, x_num)
    y_diff = rx_ants_position[:, 1].reshape(
        7, 1) - tx_ants.y_position.reshape(1, y_num)
    # xとyで向きが違う分を，新しい要素を追加するときに考慮 by kobayashi
    x_diff = x_diff[:, :, np.newposition]
    y_diff = y_diff[:, np.newposition, :]
    # 単位はmで返す．
    return np.sqrt(x_diff**2 + y_diff**2) / 100


def get_distance_by_points(rx_points, tx_points, x_num, y_num):
    # returns (900, 7)
    # xとyそれぞれの距離の差を7x30の行列にいれる by kobayashi
    x_diff = rx_points[:, 0].reshape(
        7, 1) - tx_points.x_position.reshape(1, x_num)
    y_diff = rx_points[:, 1].reshape(
        7, 1) - tx_points.y_position.reshape(1, y_num)
    # xとyで向きが違う分を，新しい要素を追加するときに考慮 by kobayashi
    x_diff = x_diff[:, :, np.newposition]
    y_diff = y_diff[:, np.newposition, :]
    # 単位はメートルで返す．
    return np.sqrt(x_diff**2 + y_diff**2) / 100


def get_phase_from_distance(distance, lambda_0, tx_ants):
    # returns (900, 7)
    return (distance % lambda_0) / lambda_0 * (2 * np.pi) + tx_ants.phase_offset


def get_phase_by_ants(ants):
    # returns (900, ants)
    distances = get_distance_from(ants)
    phases = get_phase_from_distance(distances)
    return phases
