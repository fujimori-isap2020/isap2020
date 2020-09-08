# const paramater
#
import numpy as np

# 送信アンテナの配置スペース(unit: cm)
x_start = 0
y_start = 0
x_range = 300
y_range = 300
x_num = 30
y_num = 30


# アンテナから出る電波の周波数
freq = 5.8e9
# 光速
c = 3.0e8
# 波長
lambda_0 = c / freq
# 波数
k = 2 * np.pi / lambda_0

# 送信アンテナのゲイン
tx_gain = 0
# 向き(unit:deg)
tx_orientation = 90
# パワー
tx_power = 10
# 高さ(unit:[m])
tx_antenna_hight = 0.5


# 受信アンテナのゲイン
rx_gain = 0
# y軸座標(unit:[m])
rx_ants_y_position = -1
# 数
rx_antenna_num = 7
# 高さ(unit:[m])
rx_antenna_hight = 1

# 原点
point_of_origin = 0
# 位相の原点(?)
phase_of_origin = 0
