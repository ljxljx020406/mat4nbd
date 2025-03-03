from multiband_optical_network_env import MultibandOpticalNetworkEnv
import pickle
import numpy as np
import random
import copy
import sys
import os

# 添加A文件夹的路径到sys.path
sys.path.append(os.path.abspath('../background'))

from main_functions import release_service, one_link_transmission


def evaluate_and_sort_actions(env):
    action_rewards = {}
    # print('service:', env.service.service_id)
    for wave in range(80):
        allocation, Power, path_GSNR = env.only_check_action(wave)
        if not allocation:
            reward = -100
        else:
            reward = env.only_calculate_reward(wave, Power, path_GSNR)

        action_rewards[wave] = reward
        # print(wave, reward)

    # 将字典按照奖励值从大到小排序
    sorted_action_rewards = sorted(action_rewards.items(), key=lambda item: item[1], reverse=True)

    return sorted_action_rewards[:5]

topology_file= '../topology/usnet_500service_with_fragmentation2.pkl'
with open('../service/usnet_500services_after_release2.pkl', 'rb') as f:
    service_dict = pickle.load(f)
with open('../service/usnet_500services_to_be_sorting2.pkl', 'rb') as f:
    service_to_be_sorting = pickle.load(f)

env = MultibandOpticalNetworkEnv(topology_file=topology_file, service_dict=service_dict,
                                 service_to_be_sorting=service_to_be_sorting)

observation0 = env.reset(topology_file, service_dict, service_to_be_sorting)

# print(env.service.wavelength) # 54

tmp_topology = copy.deepcopy(env.topology)
print(env.only_calculate_network_fragmentation(env.topology))
release_service(env.topology, env.service, env.service_dict)

Power = np.zeros((len(env.service.path), 80), dtype=float)
path_GSNR = np.zeros((len(env.service.path), 80), dtype=float)

for i in range((len(env.service.path) - 1)):
    u = env.service.path[i]
    v = env.service.path[i + 1]
    # 检查波长是否空闲
    if not (env.topology[u][v]['wavelength_power'][54] == 0
            or (np.isnan(env.topology[u][v]['wavelength_power'][54]))):
        # 找出所有值为零的元素的索引
        zero_indices = np.where(np.array(env.topology[u][v]['wavelength_power']) == 0.0)[0]
        # print('power:', self.topology[u][v]['wavelength_power'])
        # print('zero_incidies:', zero_indices)
        # if len(zero_indices) > 0:
        #     action = np.random.choice(zero_indices)
        # else:
        allocation = False
        reason = "wavelength occupied!"
        break
    else:
        # 检查SNR是否满足要求
        # 获取链路参数
        distance = env.topology[u][v]['length']
        channels = 80
        if i == 0:
            Power[i] = copy.deepcopy(env.topology[u][v]['wavelength_power'])
            Power[i][54] = env.service.power

        frequencies = np.concatenate(
            [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

        tmp = copy.deepcopy(env.topology[u][v]['wavelength_power'])
        tmp[54] = env.service.power

        if i != 0:
            # print('unupdate_Power[i]:', Power[i])
            Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                        range(len(Power[i]))]
            tmp = Power[i]

        Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
        path_GSNR[i] = GSNR

        if env.service.snr_requirement > GSNR[54]:
            allocation = False
            reason = "GSNR not satisfied!!"
            break
        else:
            Power[i + 1] = Power_after_transmission

            # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
            for m in range(80):
                if m != 54 and env.topology[u][v]['wavelength_service'][m] != 0:
                    tmp_service = env.service_dict.get(env.topology[u][v]['wavelength_service'][m], None)
                    # print('tmp_service:', tmp_service.snr_requirement, tmp_service.bit_rate, tmp_service.path)
                    # print(service_dict.keys())
                    # print('id:', topology[u][v]['wavelength_service'][m])
                    if tmp_service.snr_requirement >= GSNR[m]:
                        # print('tmp_service:', tmp_service.service_id, tmp_service.path)
                        allocation = False
                        reason = 'interference!'
                        outer_break = True
                        break

env.topology = tmp_topology
print(env.only_calculate_network_fragmentation(env.topology))