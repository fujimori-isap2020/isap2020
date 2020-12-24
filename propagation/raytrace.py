# Calculation of Electric Field Strength Using Ray Trace Method
# Powered by Fujimori-Lab Members (Akada, Kobayashi, Maruyama) in September 2020

import cupy as cp

# self-made function
import const


class Antennas(object):

    def __init__(self, x_position, y_position, z_position, freq, power, orientation, phase_offset):
        self.x_position = x_position                # 0 ~ 240 [cm]
        self.y_position = y_position                # 0 ~ 240 [cm]
        self.z_position = z_position                # 100 [cm]
        self.freq = freq
        self.power = power
        self.orientation = orientation
        self.lambda_0 = const.c / freq
        self.k = 2*cp.pi / self.lambda_0            # wave-number
        self.phase_offset = phase_offset

    def directivity(self, theta):
        # ダイポールの長さがλ/2であることを前提の計算->測定時には周波数を合わせること
        # Unit[倍]
        deg_gain = const.gain + 10 * \
            cp.log10(cp.abs(cp.cos(2*cp.pi / 2*cp.cos(theta)) * cp.sin(theta)))
        return 10**(deg_gain/10)


# 送信アンテナ
class TX(Antennas):
    def __init__(self, x_position, y_position, z_position, freq, power, orientation, phase_offset):
        super().__init__(x_position, y_position, z_position,
                         freq, power, orientation, phase_offset)


# 受信アンテナ
# こっち側から電力を推定する
class RX(Antennas):

    def __init__(self, x_position, y_position, z_position, freq, power, orientation, phase_offset):
        super().__init__(x_position, y_position, z_position,
                         freq, power, orientation, phase_offset)

    def calc_theta_tx(self, tx_ant):
        x, y = cp.meshgrid(tx_ant.x_position, tx_ant.y_position)
        # ブロードキャストを使って30x30に各受信点からの角度(shape=7)を格納
        rx_x = self.x_position[cp.newaxis, cp.newaxis, :]  # shape=1,1,7
        rx_y = self.y_position[cp.newaxis, cp.newaxis, :]  # shape=1,1,7
        x_diff = rx_x - x[:, :, cp.newaxis]                 # shape=30,30,7
        y_diff = rx_y - y[:, :, cp.newaxis]                 # shape=30,30,7
        theta = cp.arctan(y_diff / x_diff)
        return theta

    def calc_phi_tx(self, distance, tx_ant):
        phi = cp.arctan(distance / (tx_ant.z_position + self.z_position))
        return phi

    def get_distance_by_points(self, tx_points):
        # returns (2, 30, 30, 7)
        # meshgridで30x30のx,yの距離を格納した行列を作成(shape=30,30)
        x, y = cp.meshgrid(tx_points.x_position, tx_points.y_position)
        # ブロードキャストを使って30x30に各受信点からの距離(shape=7)を格納
        rx_x = self.x_position[cp.newaxis, cp.newaxis, :]       # shape=1,1,7
        rx_y = self.y_position[cp.newaxis, cp.newaxis, :]       # shape=1,1,7
        x_diff = rx_x - x[:, :, cp.newaxis]                     # shape=30,30,7
        y_diff = rx_y - y[:, :, cp.newaxis]                     # shape=30,30,7
        direct_z_diff = self.z_position - tx_points.z_position
        indirect_z_diff = self.z_position + tx_points.z_position
        direct_dist = cp.sqrt(x_diff**2 + y_diff**2 + direct_z_diff*2)
        indirect_dist = cp.sqrt(x_diff**2 + y_diff**2 + indirect_z_diff*2)
        # 直接波と間接波の距離を返す
        # shape=2,30,30,7
        return cp.stack([direct_dist, indirect_dist])

    def get_phase_from_distance(self, distance, tx_ants):
        # returns (2, 30, 30, 7)
        return (distance % tx_ants.lambda_0) / tx_ants.lambda_0 * (2*cp.pi) + tx_ants.phase_offset

    # 行路長差
    def path_length_difference(self, distance, ht, hr):
        delta_l = cp.sqrt((ht + hr)**2 + distance**2) - \
            cp.sqrt((ht - hr)**2 + distance**2)
        return delta_l

    # 受信電力
    def receive_power(self, tx_ant):
        # 「EEM-RTM理論説明書」3.1伝達公式参照
        # http://www.e-em.co.jp/doc/rtm_theory.pdf
        distance = self.get_distance_by_points(tx_ant) / 100    # Unit[m]
        theta = self.calc_theta_tx(tx_ant)
        phi = self.calc_phi_tx(distance[1]*100, tx_ant)
        R = self.R_horizontal(phi, const.epsilon_r)

        def E(reflection_coefficient, distance):
            E = cp.sqrt(self.directivity(theta)) * \
                cp.sqrt(self.directivity(theta)) * \
                reflection_coefficient * \
                (tx_ant.lambda_0 / (4 * cp.pi * distance)) * \
                cp.exp((-1j * tx_ant.k) * distance)
            return E

        Pr = (E(const.reflection_coefficient, distance[0]) + E(R, distance[1])) * \
            cp.exp(1j * cp.deg2rad(tx_ant.phase_offset)) * \
            cp.sqrt(tx_ant.power)

        return Pr

    # 比誘電率
    def epsilon(self, epsilon_r):
        epsilon = epsilon_r - (1j * const.sigma) / \
            (2 * cp.pi * const.freq * const.epsilon_0)
        return epsilon

    # 反射係数（垂直偏波）
    def R_vertical(self, theta, epsilon_r):
        R = (cp.sqrt(self.epsilon(epsilon_r) - (cp.sin(theta))**2) - self.epsilon(epsilon_r) * cp.cos(theta)) / \
            (cp.sqrt(self.epsilon(epsilon_r) - (cp.sin(theta))**2) +
             self.epsilon(epsilon_r) * cp.cos(theta))
        return R

    # 反射係数（水平偏波）
    def R_horizontal(self, theta, epsilon_r):
        R = (cp.cos(theta) - (cp.sqrt(self.epsilon(epsilon_r) - (cp.sin(theta))**2))) / \
            (cp.cos(theta) + cp.sqrt(self.epsilon(epsilon_r) - (cp.sin(theta))**2))
        return R
