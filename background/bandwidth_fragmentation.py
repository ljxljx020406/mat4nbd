import pickle
import random
import numpy as np

import pandas as pd
# import sys
# sys.path.append('/Users/liujiaxin/Desktop/bishe/code/ljx-optical-rl-gym-multiband-main/optical-rl-gym-multiband-main/formal code/background')

# 加载topology对象
with open('../blocking_test/usnet_with_smallservices1.pkl', 'rb') as f:
    topology = pickle.load(f)

# with open('../topology/usnet_500services5-2.pkl', 'rb') as f:
#     topology1 = pickle.load(f)
# with open('../trans_LSTM/new_topology/best_defragmentation.pkl', 'rb') as f:
#     topology1 = pickle.load(f)

# with open('../topology_with_services/24layer-4dataset-topology_defragmentationed.pkl', 'rb') as f:
#     topology2 = pickle.load(f)
# sorted_service_dict = sorted(service_dict.items(), key=lambda x: x[1].utilization)

# print(sorted_service_dict)
# for service_name, service_obj in sorted_service_dict:
#     # 在这里可以访问service_obj的属性，例如utilization
#     print(f"Service ID: {service_obj.service_id}, Utilization: {service_obj.utilization}")

# print(sorted_service_dict)

# # 从service_dict随机抽取一个元素
# random_key = random.choice(list(service_dict.keys()))
# random_service = service_dict[random_key]
#
# print(random_service.service_id, random_service.utilization)

# utilizations = []
# for u, v in topology.edges():
#     edge_data = topology[u][v]
#     utilizations.append(edge_data['wavelength_utilization'])
# # print(len(utilizations), len(utilizations[0]))
# utilizations_no_zeros = np.where(utilizations == 0, np.nan, utilizations)
# # 计算每个波长的平均利用率，忽略np.nan值
# average_utilization_per_wavelength = np.nanmean(utilizations_no_zeros, axis=0)
# average_utilization_per_wavelength_no_zeros = np.where(average_utilization_per_wavelength == 0, np.nan, average_utilization_per_wavelength)
# std_deviation = np.nanmean(average_utilization_per_wavelength_no_zeros)
# print(std_deviation)

def _calculate_network_fragmentation(topology):
    # 计算整个网络的带宽碎片度
    all_utilization_rates = []
    utilizations = []
    for u, v in topology.edges():
        edge_data = topology[u][v]
        utilizations.append(edge_data['wavelength_utilization'])
        occupied_wavelengths = np.count_nonzero(np.array(edge_data['wavelength_power']) > 0)
        if occupied_wavelengths != 0:
            total_utilization = 0
            for i in range(len(edge_data['wavelength_power'])):
                total_utilization += edge_data['wavelength_utilization'][i]
            utilization_rate = total_utilization / occupied_wavelengths
            all_utilization_rates.append(utilization_rate)
        else:
            all_utilization_rates.append(0)

    # 利用率因子：网络中所有链路的带宽利用率的平均值
    avg_utilization = np.mean(all_utilization_rates)

    # 一致性因子：网络中带宽利用率的标准差
    utilizations_no_zeros = np.where(utilizations == 0, np.nan, utilizations)
    # 计算每个波长的平均利用率，忽略np.nan值
    average_utilization_per_wavelength = np.nanmean(utilizations_no_zeros, axis=0)
    average_utilization_per_wavelength_no_zeros = np.where(average_utilization_per_wavelength == 0, np.nan,
                                                           average_utilization_per_wavelength)
    std_deviation = np.nanstd(average_utilization_per_wavelength_no_zeros)

    # 综合评估：带宽碎片度定义为利用率因子和一致性因子的加权和
    network_fragmentation = 0.9 * (1-avg_utilization) + 0.1 * std_deviation  # 简化的计算公式，权重可以调整

    return avg_utilization, std_deviation, network_fragmentation, all_utilization_rates

print(_calculate_network_fragmentation(topology))  #(0.3564168650307055, 0.08844791666666667, 0.5880696131390316)
# print(_calculate_network_fragmentation(topology1))  # (0.5304166462014416, 0.1524579613095238, 0.437870814549655)
# print(_calculate_network_fragmentation(topology2))
# _, _1, _2, list1 = _calculate_network_fragmentation(topology)
# _, _1, _2, list2 = _calculate_network_fragmentation(topology1)
# # _, _1, _2, list3 = _calculate_network_fragmentation(topology2)
# data = pd.DataFrame([list1, list2])
# excel_file = 'link_utilization5-2.xlsx'
# data.to_excel(excel_file, index=False)