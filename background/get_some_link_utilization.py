import gym
import numpy as np
import pickle
import random
import copy
import sys
import os
import itertools

topology_file = '../topology/new_usnet_500service_with_fragmentation5-2.pkl'
with open(topology_file, 'rb') as f:
    topology = pickle.load(f)
utilizations = []
all_utilization_rates = []
for u, v in topology.edges():
    edge_data = topology[u][v]
    utilizations.append(edge_data['wavelength_utilization'])
    occupied_wavelengths = np.count_nonzero(np.array(edge_data['wavelength_power']) > 0)
    total_utilization = 0
    if occupied_wavelengths != 0:
        for i in range(len(edge_data['wavelength_power'])):
            total_utilization += edge_data['wavelength_utilization'][i]
        utilization_rate = total_utilization / occupied_wavelengths
        all_utilization_rates.append(utilization_rate)
    else:
        all_utilization_rates.append(0)

# 利用率因子：网络中所有链路的带宽利用率的平均值
avg_utilization = np.mean(all_utilization_rates)