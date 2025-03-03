from main_functions import (new_service, naive_RWA, first_fit, release_service, check_utilization,
                            _numba_one_link_transmission, one_link_transmission, get_Pi_z, calculate_ASE_noise)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import time
import pandas
import openpyxl
import copy
import numba
from numba import jit, njit, prange
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

with open('../topology/usnet_topology_1path.h5', 'rb') as f:
    topology = pickle.load(f)
with open('../service/usnet_100erl_1w.pkl', 'rb') as f:
    services = pickle.load(f)

import pandas as pd

row = 0
service_dict = {}
for service in services.values():
    time0 = service.arrival_time
    print('id/src/dst/bitrate:', service.service_id, service.source_id, service.destination_id, service.bit_rate)
    static_dict = copy.deepcopy(service_dict)
    for i in static_dict.keys():
        if static_dict[i].holding_time <= time0:
            release_service(topology, static_dict[i], service_dict)
    path, wavelength, reason = naive_RWA(topology, service, service_dict)
    if path == None:
        print('blocking!!!!!!!')
        # 将每个reason转换为一个DataFrame并转置为一行
        df1 = pd.DataFrame([reason])  # 将reason转换为一个包含一行的DataFrame

        # 使用 openpyxl 引擎来实现追加写入，并避免重复写入表头
        with pd.ExcelWriter("阻塞原因测试.xlsx", mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # 将数据写入 Excel 的当前行，避免写入列标题
            df1.to_excel(writer, index=False, header=False, sheet_name='Sheet1', startrow=row)

        row += 1  # 更新行号，确保每次数据写入下一行
print('blocking_ratio:', row/len(service_dict))