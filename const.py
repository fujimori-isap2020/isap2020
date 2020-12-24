# const paramater
#
import cupy as cp

# antenna position example
#
# --|
#   |         __ Tx Antenna
#   |        /
#   |       x
# 2 |
# 4 |                   x
# 0 |
#   |
#   |           x
#   |
# --+------------------------------
# 1 |            240              |
# 0 |
# 0 |
# --x    x    x    x    x    x    x
#   |         Rx Antenna


# 原点
point_of_origin = 0
# 位相の原点(?)
phase_of_origin = 0

# アンテナから出る電波の周波数 [Hz]
freq = 5.80e9
# 光速 [m/s]
c = 2.99792e8
# 波長 [m]
lambda_0 = c / freq
# 波数
k = 2 * cp.pi / lambda_0
# 反射係数
reflection_coefficient = 1
# 空間インピーダンス
eta = 120 * cp.pi

# ダイポールのインピーダンス
Zr = 73.1 + 42.5j
# ダイポールのゲイン [dBi]
gain = 2.14
# ダイポールのエレメント長 [mm]
dipole_element_length = 76
# ダイポールの実効長 [mm]
dipole_element_le = lambda_0 / cp.pi * 10e3

# 真空の誘電率[F/m]
epsilon_0 = 8.854e-12
# コンクリートの比誘電率（5.8GHzのときではなく1GHz時）
epsilon_r = 6.5
# 導電率[S/m]（1GHz）
sigma = 0.1

# 向き[deg]
tx_orientation = 0
# 入力電力[mW]
tx_power = 1
# 高さ[cm]
tx_antenna_hight = 100
# 送信アンテナの配置スペース[cm]
x_start = 0
y_start = 0
x_range = 240
y_range = 240
x_num = 30
y_num = 30
tx_z_position = 100
# 送信アンテナの配置グリッド
x1_grid = cp.linspace(x_start, x_range, x_num)
y1_grid = cp.linspace(y_start, y_range, y_num)
x2_grid = cp.linspace(x_start, x_range, x_num)
y2_grid = cp.linspace(y_start, y_range, y_num)


# 向き
rx_orientation = 0
# y軸座標[cm]
rx_ants_y_position = -100
# 測定点数
rx_antenna_num = 7
# 高さ[cm]
rx_antenna_hight = 100
# x軸の位置
rx_ants_x_position = cp.linspace(
    point_of_origin, x_range, rx_antenna_num)
# y軸の位置
rx_ants_y_position = cp.full(
    rx_antenna_num, rx_ants_y_position)
# 受信アンテナのx,y軸の位置
rx_ants_position = cp.stack(
    [rx_ants_x_position, rx_ants_y_position], 1)
