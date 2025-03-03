import numpy as np
import pandas as pd

from main_functions import one_link_transmission
from dummylight_transmission import apply_slope

import math

def watt_to_dbm(power):
    result = []
    for watt in power:
        # print(watt)
        result.append(10 * math.log10(watt * 1000))
    return result
def dbm2watt(value):
    '''
    dBm -> W
    '''
    return 10 ** ((value - 30) / 10)

Power = np.zeros(80, dtype=float)
# Power[0:80] = 0.0031622776602
Power[0:80] = 5
# Power[3:6] = 0.0031622776602
# Power[15] = 0.0031622776602
# Power[42:50] = 0.0031622776602
# Power[51:60] = 0.0031622776602
distance = 800000
channels = 80
frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
l_band_indices = np.arange(0, 40)  # C波段为前40个波长
c_band_indices = np.arange(40, 80)  # L波段为后40个波长

# adjusted_power = apply_slope(Power.copy(), l_band_indices, c_band_indices)
# print('adjusted_power:', adjusted_power)
adjusted_power = dbm2watt(Power)
print('adjusted_power:', adjusted_power)
Power, GSNR = one_link_transmission(distance, channels, adjusted_power, frequencies)
print('power:', Power)
print('gsnr:', GSNR)
# power_in_dbm = watt_to_dbm(Power[0:80])
# # 将数据转换为DataFrame
# df = pd.DataFrame(power_in_dbm)
# print(power_in_dbm)
df = pd.DataFrame(GSNR)
# 将DataFrame写入Excel文件
df.to_excel('dummy_GSNR_notilt.xlsx', index=False, engine='openpyxl')