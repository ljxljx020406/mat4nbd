from main_functions import (new_service, naive_RWA, release_service, check_utilization,
                            _numba_one_link_transmission, one_link_transmission, get_Pi_z, calculate_ASE_noise)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import pandas
import openpyxl
import numba
from numba import jit, njit, prange
from concurrent.futures import ThreadPoolExecutor


distance = int(800000)
channels = 10
Power = np.ones(80, dtype=float) * 0.003
frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
RefLambda = 1575e-9
num_of_spans = int(np.ceil(distance / 100e3))
RefFreq = 190.5e12
fi = np.array([(f - RefFreq) for f in frequencies])
fi = fi.reshape(-1, 1)

Bch = 150e9 * np.ones((channels, num_of_spans))
Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans))

result3 = _numba_one_link_transmission(distance, channels, Power, frequencies)
print('get Pi finished!', result3)
services_processed_since_reset = 1
max_len_path = 0
# topology = get_topology('/topologies/usnet.txt', 'USNET')

with open('../topology/usnet_topology.h5', 'rb') as f:
    topology = pickle.load(f)

service_dict = {}

while services_processed_since_reset < 150:
    service = new_service(topology, services_processed_since_reset)
    print('id/src/dst/bitrate:',service.service_id, service.source_id, service.destination_id, service.bit_rate)
    path, wavelength = naive_RWA(topology, service, service_dict)

    if path != None:
        services_processed_since_reset += 1
        check_utilization(topology, path, service_dict)
        if len(path) > max_len_path:
            max_len_path = len(path)
            print('===========================max_len_path:', max_len_path, '===============================')

for u, v, attributes in topology.edges(data=True):
    utilization = attributes.get('wavelength_utilization', [])
    power = attributes.get('wavelength_power', [])
    snr = attributes.get('wavelength_SNR', [])
    service_id = attributes.get('wavelength_service', [])
    # print('power_length:', len(power))
    for i in range(len(power)):
        if utilization[i] == 0 and power[i] != 0:
            print("1异常！：", "链路:", f"{u}-{v}", "波长:", i, "利用率:", utilization[i], "功率:",
                  power[i], "SNR:", snr[i])
        if utilization[i] != 0 and power[i] == 0:
            print("2异常！：", "链路:", f"{u}-{v}", "波长:", i, "利用率:", utilization[i], "功率:",
                  power[i], "SNR:", snr[i])
        # if service_id[i] != 0 and power[i] == 0:
        #     print("异常！：", "链路:", f"{u}-{v}", "波长:", i, "利用率:", utilization[i], "功率:",
        #           power[i], "SNR:", snr[i], 'service_id:', service_id[i])
# 计算并设置每条边非零元素的平均利用率
# for u, v, data in topology_copy.edges(data=True):
#     utilization = data['wavelength_utilization']
#     non_zero_avg = np.mean(utilization[utilization != 0]) if np.any(utilization != 0) else 0
#     data['avg_utilization'] = non_zero_avg
#     print("边的平均利用率为：", non_zero_avg)

# # 为每条链路添加非零元素平均利用率标签
# edge_labels = {(u, v): f'{data["avg_utilization"]:.2f}' for u, v, data in topology.edges(data=True)}
# nx.draw_networkx_edge_labels(topology, pos, edge_labels=edge_labels)
#
# # 显示图形
# plt.show()

# service_list_after_release = service_list.copy()

# 保存topology对象到文件
with open('../topology/1030usnet_150services2.pkl', 'wb') as f:
    pickle.dump(topology, f)

with open('../service/1030usnet_150service_dict2.pkl', 'wb') as f:
    pickle.dump(service_dict, f)

# 提取数据
data = []
for u, v, attributes in topology.edges(data=True):
    utilization = attributes.get('wavelength_utilization', [])
    power = attributes.get('wavelength_power', [])
    snr = attributes.get('wavelength_SNR', [])
    id = attributes.get('wavelength_service', [])
    for wavelength in range(len(power)):
        data.append({
            "链路": f"{u}-{v}",
            "波长": wavelength,
            "利用率": utilization[wavelength],
            "功率": power[wavelength],
            "SNR": snr[wavelength],
            "id": id[wavelength]
        })
        # if utilization[wavelength] == 0 and power[wavelength] != 0:
        #     print("异常！：", "链路:", f"{u}-{v}", "波长:", wavelength, "利用率:", utilization[wavelength], "功率:", power[wavelength], "SNR:", snr[wavelength])
# 创建DataFrame
df1 = pandas.DataFrame(data)

# for service_id, service in service_dict.items():
#     # print('service_wavelength:', service.wavelength)
#     if service.wavelength < 40:
#
#         # print('service_id:', service.service_id, service_list_after_release[service.service_id - 1].service_id)
#         release_service(topology, service, service_dict)
#         # continue

to_delete = []
for service_id, service in service_dict.items():
    if service.wavelength < 20:
        to_delete.append(service_id)
print('to_delete:', to_delete)
for service_id in to_delete:
    service = service_dict[service_id]
    # print('service:', service.bit_rate)
    release_service(topology, service, service_dict)

# release_influenced_services(topology, service_dict)

# 保存service_dict到文件
with open('../service/1030usnet_150services_after_release2.pkl', 'wb') as f:
    pickle.dump(service_dict, f)

# 保存topology对象到文件
with open('../topology/1030usnet_150service_with_fragmentation2.pkl', 'wb') as f:
    pickle.dump(topology, f)

# 提取数据
data = []
for u, v, attributes in topology.edges(data=True):
    utilization = attributes.get('wavelength_utilization', [])
    power = attributes.get('wavelength_power', [])
    snr = attributes.get('wavelength_SNR', [])
    id = attributes.get('wavelength_service', [])
    for wavelength in range(len(utilization)):
        data.append({
            "利用率": utilization[wavelength],
            "功率": power[wavelength],
            "SNR": snr[wavelength],
            "id": id[wavelength]
        })
# 创建DataFrame
df2 = pandas.DataFrame(data)

# 创建ExcelWriter对象
with pandas.ExcelWriter("1030usnet_150services_链路数据2.xlsx") as writer:
    # 将第一批数据写入工作表
    df1.to_excel(writer, index=False, sheet_name='Sheet1')

    # 计算第一批数据占用的列数
    num_cols_df1 = len(df1.columns)

    # 将第二批数据写入工作表，从足够远的列开始以避免覆盖
    df2.to_excel(writer, index=False, sheet_name='Sheet1', startcol=num_cols_df1 + 1)  # 加1为了留出空列

print("---------------------------")
# 计算并设置每条边非零元素的平均利用率
for u, v, data in topology.edges(data=True):
    utilization = data['wavelength_utilization']
    non_zero_avg = np.mean(utilization[utilization != 0]) if np.any(utilization != 0) else 0
    data['avg_utilization'] = non_zero_avg
    print("边的平均利用率为：", non_zero_avg)

print('max_len_path:', max_len_path)