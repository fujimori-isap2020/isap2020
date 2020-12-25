# this program is made for sdc in isap2020
# transmitter A: 300-320MHz, -50dBm, lambda/2 dipole, horizontal, 1-3antennas, CW
# transmitter B: 2.402-2.480GHz, 0dBm, Small antenna, unknown pole, 1-3antennas, ble beacon
# transmitter C: 5.012-5.025GHz, 10dBm, Monopole antenna, Vertical, 1-2, CW
# Powered by Fujimori-Lab Members (Akada, Kobayashi, Maruyama) in September 2020
#
# ---- Memo ----
# 20201118: classで距離を測定するやつを作っていたが遅くなりそうなので関数にした
#           見た目より高速化を優先する
# 20201202  速度よりとりあえずclassで作ることに
# 20201222  送信・受信ともにダイポールアンテナの指向性を持つこと前提で演算
#           送信アンテナの方向は0[deg]で固定，高さも固定（100[cm]）
#           ダイポールアンテナはλ/2の周波数で入力すること（指向性をそれで決めてる）
#           定数類は全部const.pyに入っている
#           水平偏波（TM波）前提
#           全部ndarrayにブチ込んでcupyで高速処理よ
# 20201223  メイン関数は小さく


import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# self-made function
from propagation import raytrace as rt
import const

# Unified Memoryを使う
# pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
# cp.cuda.set_allocator(pool.malloc)


def calc_antennas_vector(phase_diff):

    # 送信アンテナのインスタンス作成
    # txはx,yのgridに対してとりうる点をすべて保持する．（30x30)
    # アンテナの高さと方向は固定
    # TM波（水平偏波）
    tx_ant1 = rt.TX(const.x1_grid, const.y1_grid, const.tx_antenna_hight,
                    const.freq, const.tx_power, const.tx_orientation,
                    const.phase_of_origin)
    tx_ant2 = rt.TX(const.x2_grid, const.y2_grid, const.tx_antenna_hight,
                    const.freq, const.tx_power, const.tx_orientation, (phase_diff/180)*2*cp.pi)

    # 受信アンテナのインスタンス作成
    # rxもx,yのgridに対して取りうる点をすべて保持する（今回は7点のみx:0,50...,y:100）
    # アンテナの高さと方向は固定
    rx_ants = rt.RX(const.rx_ants_x_position, const.rx_ants_y_position,
                    const.rx_antenna_hight, const.freq, const.gain,
                    const.rx_orientation, const.phase_of_origin)

    # 2波モデルを使って受信点でのパワーを計算
    # power_rxs.shape = (30, 30, 7)
    power_rx1 = rx_ants.receive_power(tx_ant1)
    power_rx2 = rx_ants.receive_power(tx_ant2)

    power_rxs = power_rx1[:, :, cp.newaxis, cp.newaxis, :] + \
        power_rx2[cp.newaxis, cp.newaxis, :, :, :]

    # unit[dBm]
    power_rxs = 10 * cp.log10(cp.abs(power_rxs**2))

    return power_rxs


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


# シリアライズしたデータの順番
# shape = (30, 30, 30, 30, 7)
# (x1_position, y1_position, x2_position, y2_position, rx_power)
if __name__ == '__main__':
    deg_start = sys.argv[1]
    deg_stop = sys.argv[2]
    print(f'processing {deg_start} from {deg_stop}')
    # temp = cp.empty([int(deg_stop)+1, const.x_num, const.y_num, const.x_num,
    #                  const.y_num, const.rx_antenna_num])
    os.makedirs("traning_data", exist_ok=True)
    # rangeはendpointが含まれないので，stop+1している
    for deg in tqdm(range(int(deg_start), int(deg_stop)+1, 1)):
        cp.savez("/tmp/training_data/training_data_"+str(deg),
                 calc_antennas_vector(deg))
