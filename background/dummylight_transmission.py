import numpy as np
import matplotlib.pyplot as plt
from numpy import log10

def lin2db(value):
    '''
    ln -> dB
    '''
    return 10 * log10(value)
def watt2dbm(value):
    '''
    W -> dBm
    '''
    return lin2db(value * 1e3)
def dbm2watt(value):
    '''
    dBm -> W
    '''
    return 10 ** ((value - 30) / 10)


# 设置波长范围
num_channels = 80
c_band_range = np.linspace(1530, 1565, 40)  # C波段波长范围 (假设前40个是C波段)
l_band_range = np.linspace(1565, 1625, 40)  # L波段波长范围 (后40个是L波段)

# 初始功率分布 (单位: dBm)，可以是一个初始值列表
# initial_power = np.random.uniform(-10, 5, num_channels)  # 随机生成-10到5 dBm之间的功率值
initial_power = np.ones(num_channels) * 5

# 设定预倾斜系数，C波段的功率比L波段高
def apply_slope(initial_power, l_band_indices, c_band_indices, slope_factor=0.19):
    """应用一个线性预倾斜因子，C波段功率较高，L波段较低"""

    # 对C波段应用正向倾斜（增大功率）
    for i in l_band_indices:
        # initial_power[i] += slope_factor * (40 - l_band_indices[i])
        initial_power[i] -= slope_factor * (40 - l_band_indices[i])

    # 对L波段应用反向倾斜（减小功率）
    for i in c_band_indices:
        # initial_power[i] -= slope_factor * (i - c_band_indices[0])
        initial_power[i] += slope_factor * (c_band_indices[i-40] - 40)

    return initial_power


# 应用预倾斜
l_band_indices = np.arange(0, 40)  # C波段为前40个波长
c_band_indices = np.arange(40, 80)  # L波段为后40个波长

# 调用预倾斜函数
adjusted_power = apply_slope(initial_power.copy(), l_band_indices, c_band_indices)

for i in range(len(adjusted_power)):
    adjusted_power[i] = dbm2watt(adjusted_power[i])
for i in range(len(initial_power)):
    initial_power[i] = dbm2watt(initial_power[i])
# 可视化初始功率与预倾斜后的功率
plt.figure(figsize=(10, 6))
plt.plot(range(num_channels), initial_power, label='Initial Power (dBm)', linestyle='--', marker='o')
plt.plot(range(num_channels), adjusted_power, label='Adjusted Power with Pre-slope (dBm)', marker='x')
plt.xlabel('Wavelength Channel')
plt.ylabel('Power (dBm)')
plt.title('Power Distribution with and without Pre-slope')
plt.legend()
plt.grid(True)
plt.show()

# 返回经过预倾斜后的功率值
# print(adjusted_power)
