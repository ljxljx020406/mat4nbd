import time

import gym
import numpy as np
import pickle
import random
import copy
import sys
import os
import itertools

sys.path.append(os.path.abspath('../background'))

from main_functions import release_service, one_link_transmission, naive_RWA, check_RWA
import numba
from numba import jit, njit, prange
from concurrent.futures import ThreadPoolExecutor


class MultibandOpticalNetworkEnv(gym.Env):
    def __init__(self, topology, service_dict, service_to_be_sorting, num_agents, blocked_service):
        super(MultibandOpticalNetworkEnv, self).__init__()
        # 加载拓扑
        # with open(topology_file, 'rb') as f:
        #     self.topology = pickle.load(f)
        self.topology = topology
        # self.sorted_service_dict = sorted(service_dict.items(), key=lambda x: x[1].utilization)
        self.service_dict = service_dict
        self.service_to_be_sorting = service_to_be_sorting
        self.num_agents = num_agents
        self.blocked_service = blocked_service

        self.service_ids = list(self.service_to_be_sorting.keys())
        # self.observation_space = gym.spaces.Box(low=0, high=1+1e-6, shape=(81,2), dtype=float)
        # self.share_observation_space = self.observation_space
        self.observation_space = [gym.spaces.Box(low=0, high=1+1e-6, shape=(162,), dtype=float)
                                  for n in range(self.num_agents)]
        self.share_observation_space = self.observation_space.copy()
        # self.action_space = gym.spaces.Box(low=0, high=79, shape=(1,), dtype=int)
        self.action_space = [gym.spaces.Discrete(80) for n in range(self.num_agents)]

        self.initial_topology = copy.deepcopy(self.topology)

        self.episode_over = True

        self.active_masks = np.ones((self.num_agents, 1), dtype=float)
        self.active_masks[len(self.service_to_be_sorting):] = 0


    # def seed(self, seed):
    #     '''
    #     设置随机种子以确保实验的可重复性
    #     '''
    #     random.seed(seed)
    #     np.random.seed(seed)

    def reset(self):
        # with open(topology_file, 'rb') as f:
            # self.topology = pickle.load(f)
        # self.topology = topology
        # # self.sorted_service_dict = sorted(service_dict.items(), key=lambda x: x[1].utilization)
        # self.service_dict = service_dict
        # self.service_to_be_sorting = service_to_be_sorting  # 已按重要程度降序排列
        self.service_ids = list(self.service_to_be_sorting.keys())

        src = self.blocked_service.source_id
        dst = self.blocked_service.destination_id
        self.blocked_service.path = self.topology.graph['ksp'][str(src), str(dst)][0].node_list

        obs = self.get_observation()
        shared_obs = obs
        available_actions = np.ones((self.num_agents, 80))

        for i in self.service_ids:
            current_service = self.service_to_be_sorting[i]
            index = self.service_ids.index(i)
            for j in range((len(current_service.path)-1)):
                u = current_service.path[j]
                v = current_service.path[j + 1]
                for k in range(80):
                    if self.topology[u][v]['wavelength_service'][k] != 0:
                        available_actions[index][k] = 0

        # active_masks = np.ones((self.num_agents, 1), dtype=float)
        # active_masks[len(self.service_to_be_sorting):] = 0

        return obs, shared_obs, available_actions

    def get_observation(self):
        observation = []
        for i in self.service_ids:
            tmp_observation = np.zeros((81, 2))
            current_service = self.service_to_be_sorting[i]
            index = self.service_ids.index(i) + 1
            # print('current_service:', current_service.service_id)
            tmp_observation[0][0] = index/self.num_agents
            tmp_observation[0][1] = current_service.bit_rate / 900
            for j in range((len(current_service.path)-1)):
                u = current_service.path[j]
                v = current_service.path[j + 1]
                # print('u,v', u, v)
                for k in range(80):
                    # if int(self.topology[u][v]['wavelength_service'][k]) != 0:
                        # print(int(self.topology[u][v]['wavelength_service'][k]), self.service_ids)
                    if int(self.topology[u][v]['wavelength_service'][k]) in self.service_ids:
                        tmp_index = self.service_ids.index(int(self.topology[u][v]['wavelength_service'][k])) + 1
                        # print('tmp_id:', tmp_index)
                        if tmp_observation[k+1][0] == 0:
                            tmp_observation[k+1][0] = tmp_index / self.num_agents
                            # print('tmp_obs:', tmp_observation[k+1][0])
                            tmp_observation[k+1][1] = self.topology[u][v]['wavelength_bitrate'][k] / 900
                        elif tmp_observation[k + 1][0] != 0 and self.topology[u][v]['wavelength_bitrate'][k]/900 > tmp_observation[k + 1][1]:
                            tmp_observation[k + 1][0] = tmp_index / self.num_agents
                            # print('tmp_obs:', tmp_observation[k + 1][0])
                            tmp_observation[k + 1][1] = self.topology[u][v]['wavelength_bitrate'][k] / 900
                    elif ((int(self.topology[u][v]['wavelength_service'][k]) != 0)
                          and (int(self.topology[u][v]['wavelength_service'][k]) not in self.service_ids)):
                        tmp_observation[k + 1][0] = 0
                        tmp_observation[k + 1][1] = self.topology[u][v]['wavelength_bitrate'][k] / 900

            observation.append(tmp_observation)

        while len(observation) < self.num_agents:
            observation.append(np.zeros((81, 2)))

        observation_array = np.array(observation)
        observation_array = observation_array.reshape((self.num_agents, 162))

        return observation_array

    def check_action(self, action, topology, service, service_dict):
        '''
        耗时0.0x，好像还是过长？
        '''
        release_service(topology, service, service_dict)

        allocation = True
        outer_break = False
        Power = np.zeros((len(service.path), 80), dtype=float)
        path_GSNR = np.zeros((len(service.path), 80), dtype=float)

        for i in range((len(service.path) - 1)):
            if outer_break:
                break
            u = service.path[i]
            v = service.path[i + 1]
            # 检查波长是否空闲
            if not (topology[u][v]['wavelength_power'][action] == 0
                    or (np.isnan(topology[u][v]['wavelength_power'][action]))):
                allocation = False
                reason = "wavelength occupied!"
                break
            else:
                # 检查SNR是否满足要求
                # 获取链路参数
                distance = topology[u][v]['length']
                channels = 80
                Power[i] = copy.deepcopy(topology[u][v]['wavelength_power'])
                Power[i][action] = service.power

                frequencies = np.concatenate(
                    [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

                tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
                tmp[action] = service.power
                tmp = np.array(tmp)

                Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                path_GSNR[i] = GSNR

                if service.snr_requirement > GSNR[action]:
                    allocation = False
                    reason = "GSNR not satisfied!!"
                    break
                else:
                    # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                    for m in range(80):
                        if m != action and topology[u][v]['wavelength_service'][m] != 0:
                            tmp_service = service_dict.get(topology[u][v]['wavelength_service'][m], None)
                            if tmp_service.snr_requirement >= GSNR[m]:
                                allocation = False
                                reason = 'interference!'
                                outer_break = True
                                break

        if allocation:
            # print('Path:', path.node_list, 'Wavelength:', j)
            service_dict[service.service_id] = service
            service.path = service.path
            service.wavelength = action
            for i in range(len(service.path) - 1):
                u = service.path[i]
                v = service.path[i + 1]
                topology[u][v]['wavelength_power'] = Power[i]
                topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                topology[u][v]['wavelength_bitrate'][action] = service.bit_rate
                if topology[u][v]['wavelength_SNR'][action] > 24.6:
                    capacity = 900
                elif topology[u][v]['wavelength_SNR'][action] > 21.6:
                    capacity = 750
                elif topology[u][v]['wavelength_SNR'][action] > 18.6:
                    capacity = 600
                elif topology[u][v]['wavelength_SNR'][action] > 16:
                    capacity = 450
                elif topology[u][v]['wavelength_SNR'][action] > 12:
                    capacity = 300
                else:
                    capacity = 150
                # capacity = 800 if topology[u][v]['wavelength_SNR'][j] > 26.5 else 400
                topology[u][v]['wavelength_utilization'][action] = service.bit_rate / capacity
                topology[u][v]['wavelength_service'][action] = service.service_id

                # # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                for wave in range(80):
                    if wave != action and topology[u][v]['wavelength_service'][wave] != 0:
                        service_id = topology[u][v]['wavelength_service'][wave]
                        tmp_service = service_dict.get(service_id, None)
                        # print('tmp_service:', service_id)
                        if topology[u][v]['wavelength_SNR'][wave] > 24.6:
                            capacity = 900
                        elif topology[u][v]['wavelength_SNR'][wave] > 21.6:
                            capacity = 750
                        elif topology[u][v]['wavelength_SNR'][wave] > 18.6:
                            capacity = 600
                        elif topology[u][v]['wavelength_SNR'][wave] > 16:
                            capacity = 450
                        elif topology[u][v]['wavelength_SNR'][wave] > 12:
                            capacity = 300
                        else:
                            capacity = 150
                        topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity

        return allocation, Power, path_GSNR

    def calculate_network_fragmentation(self, tmp_topology):
        # 计算整个网络的带宽碎片度 -> 计算涉及到被阻塞业务的链路的总带宽碎片度
        all_utilization_rates = []
        utilizations = []
        for i in range(len(self.blocked_service.path)-1):
            u = self.blocked_service.path[i]
            v = self.blocked_service.path[i+1]
            edge_data = tmp_topology[u][v]
            utilizations.append(edge_data['wavelength_utilization'])
            occupied_wavelengths = np.count_nonzero(np.array(edge_data['wavelength_power']) > 0)
            total_utilization = 0
            for i in range(len(edge_data['wavelength_power'])):
                total_utilization += edge_data['wavelength_utilization'][i]
            if occupied_wavelengths != 0:
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
        network_fragmentation = 0.9 * (1 - avg_utilization) + 0.1 * std_deviation  # 简化的计算公式，权重可以调整
        # print(avg_utilization, std_deviation)

        return network_fragmentation

    def calculate_reward(self, new_topology):
        origin_frag = self.calculate_network_fragmentation(self.topology)
        # print('origin_frag:', origin_frag)
        current_frag = self.calculate_network_fragmentation(new_topology)
        # print('current_frag:', current_frag)
        # print(self.topology == new_topology)
        return (origin_frag - current_frag) * 100

    def make_step(self, actions, current_step, episode_length):
        blocked_allocation = False
        rewards = np.zeros((self.num_agents, 1))
        available_actions = np.ones((self.num_agents, 80))
        dones = np.ones(self.num_agents, dtype=bool)
        dones[:len(self.service_to_be_sorting)] = False
        infos = {}
        for i in self.service_ids:
            index = self.service_ids.index(i)
            current_service = self.service_to_be_sorting[i]
            # print(i, 'current_service:', current_service.service_id, current_service.wavelength, actions[index])
            tmp_topology = copy.deepcopy(self.topology)
            tmp_service_dict = copy.deepcopy(self.service_dict)
            allocation, path, GSNR = self.check_action(int(actions[index]), tmp_topology, current_service, tmp_service_dict)
            if allocation:
                # print('allocation!')
                self.service_dict = tmp_service_dict
                rewards[index] = self.calculate_reward(tmp_topology)
                # print('reward:', rewards)
                self.topology = tmp_topology
            else:
                # print('not allocation!')
                rewards[index] = -1

        if current_step == episode_length:
            dones = np.ones(self.num_agents, dtype=bool)
        path, wavelength, reason = check_RWA(self.topology, self.blocked_service, self.service_dict)
        if wavelength != None:
            blocked_allocation = True
            dones = np.ones(self.num_agents, dtype=bool)

        obs = self.get_observation()
        shared_obs = obs

        for i in self.service_ids:
            current_service = self.service_to_be_sorting[i]
            index = self.service_ids.index(i)
            for j in range((len(current_service.path)-1)):
                u = current_service.path[j]
                v = current_service.path[j + 1]
                for k in range(80):
                    if self.topology[u][v]['wavelength_service'][k] != 0 \
                            and self.topology[u][v]['wavelength_service'][k] != current_service.service_id:
                        available_actions[index][k] = 0
        return obs, shared_obs, rewards, dones, infos, available_actions, blocked_allocation