from main_functions import one_link_transmission
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import time
import pandas
import openpyxl
import numba
from numba import jit, njit, prange
from concurrent.futures import ThreadPoolExecutor


distance = 1000e3
channels = 80
Power = 0.0031622776602 * np.ones(channels)
Power[6] = 0
Power[10:30] = 0
Power[40:50] = 0
frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

Power1, GSNR1 = one_link_transmission(distance, channels, Power, frequencies)
print('GSNR:', GSNR1)

# Power = 0.0031622776602 * np.ones(channels)
# Power[5] = 0
# Power[10:30] = 0
# Power[40:50] = 0
# Power2, GSNR2 = one_link_transmission(distance, channels, Power, frequencies)
#
#
# distance = 1000e3
# channels = 80
# Power = 0.0031622776602 * np.ones(channels)
# Power[6] = 0
# Power[11:31] = 0
# Power[40:50] = 0
# frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
#
# Power3, GSNR3 = one_link_transmission(distance, channels, Power, frequencies)
#
# distance = 1000e3
# channels = 80
# Power = 0.0031622776602 * np.ones(channels)
# Power[5] = 0
# Power[11:31] = 0
# Power[40:50] = 0
# frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
#
# Power4, GSNR4 = one_link_transmission(distance, channels, Power, frequencies)
# print((Power2-Power1)+(Power3-Power1) == (Power4-Power1))
# print((GSNR2-GSNR1)+(GSNR3-GSNR1) == (GSNR4-GSNR1))
# Power = 0.0031622776602 * np.ones(channels)
# Power[5] = 0
# Power[10:30] = 0
# Power[40:60] = 0
# Power, GSNR3 = one_link_transmission(distance, channels, Power, frequencies)
# # print('GSNR:', GSNR2)
# print(GSNR3-GSNR1)
# df1 = pandas.DataFrame(GSNR1)
# df2 = pandas.DataFrame(GSNR2)
# df3 = pandas.DataFrame(GSNR3)
# with pandas.ExcelWriter("GSNR测试.xlsx") as writer:
#     # 将第一批数据写入工作表
#     df1.to_excel(writer, index=False, sheet_name='Sheet1')
#     df2.to_excel(writer, index=False, sheet_name='Sheet1', startcol=1)  # 加1为了留出空列
#     df3.to_excel(writer, index=False, sheet_name='Sheet1', startcol=2)  # 加1为了留出空列

# time1 = time.time()
# result = np.zeros(channels)
# for i in range(channels):
#     if Power[i] == 0:
#         Power[i] = 0.0031622776602
#         Power, GSNR = one_link_transmission(distance, channels, Power, frequencies)
#         result[i] = GSNR[i]
#         Power[i] = 0
# print(result)
# time2 = time.time()
# print('time:', time2-time1)