import gym
import numpy as np
import pickle
import random
import copy
import sys
import os
import itertools

sys.path.append(os.path.abspath('../background'))

from main_functions import release_service, one_link_transmission

class MultibandOpticalNetworkEnv(gym.Env):
    def __init__(self, topology, service_dict, service_to_be_sorting):
        super(MultibandOpticalNetworkEnv, self).__init__()
        # 加载拓扑
        # with open(topology_file, 'rb') as f:
        #     self.topology = pickle.load(f)
        self.topology = topology
        # self.sorted_service_dict = sorted(service_dict.items(), key=lambda x: x[1].utilization)
        self.service_dict = service_dict
        self.service_to_be_sorting = service_to_be_sorting
        self.service_ids = list(self.service_to_be_sorting.keys())
        # self.service_ids.sort()  # 升序
        self.initial_topology = copy.deepcopy(self.topology)

        self.service = None
        self.current_service = 0

        self.observation_space = gym.spaces.Box(low=0, high=800, shape=(7062,), dtype=np.float32)
        self.observation = None
        self.action_space = gym.spaces.Discrete(80)

        self.numEdges = len(self.topology.edges())
        self.numNodes = len(self.topology.nodes())

        self.episode_over = True
        self.tmp_reward = 0
        self.current_reward = 0

    def seed(self, seed):
        '''
        设置随机种子以确保实验的可重复性
        '''
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, topology, service_dict, service_to_be_sorting):
        # with open(topology_file, 'rb') as f:
            # self.topology = pickle.load(f)
        self.topology = topology
        # self.sorted_service_dict = sorted(service_dict.items(), key=lambda x: x[1].utilization)
        self.service_dict = service_dict
        self.service_to_be_sorting = service_to_be_sorting
        self.service_ids = list(self.service_to_be_sorting.keys())
        # random.shuffle(self.service_ids)  # 随机
        # self.service_ids.sort(reverse=True)  # 降序
        # self.service_ids.sort()  # 升序
        self.current_service = 0

        self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        # print('service:', self.service.service_id)
        self.current_service += 1
        self.tmp_reward = 0
        self.current_reward = self.get_reward(True)

        return self.get_observation()

    def get_observation(self):
        observation = []
        service_state = []
        # for i in range(len(self.service.path) - 1):
        #     u = self.service.path[i]
        #     v = self.service.path[i + 1]
        #     observation.append(self.topology[u][v]['wavelength_SNR'])
        # while len(observation) < 8:
        #     observation.append(np.zeros(80, dtype=float))
        for u,v,data in self.topology.edges(data=True):
            observation.append(data['wavelength_SNR'])
            observation.append(data['wavelength_service'])

        observation_array = np.array(observation)
        # observation_array = observation_array.flatten()
        # observation_array = np.append(observation_array, self.service.bit_rate)
        # # 将一维数组转换为1x3361的二维数组
        # observation_array = observation_array.reshape(1, -1)
        tmp_service_state = []
        for service_id, service in self.service_to_be_sorting.items():
            tmp_service_state.append(service_id)
        while len(tmp_service_state) < 50:
            tmp_service_state.append(0)
        tmp_service_state = tmp_service_state[:50]
        service_state.append(tmp_service_state)
        tmp_service_state = []
        for service_id, service in self.service_to_be_sorting.items():
            if service.bit_rate <= 400:
                tmp_service_state.append(0)
            else:
                tmp_service_state.append(1)
        while len(tmp_service_state) < 50:
            tmp_service_state.append(0)
        tmp_service_state = tmp_service_state[:50]
        service_state.append(tmp_service_state)

        service_state.append([self.service.service_id])
        if self.service.bit_rate <= 400:
            service_state.append([0])
        else:
            service_state.append([1])

        link_SNR = self.cal_link_SNR()
        service_state.append(link_SNR.tolist())
        # 展平 service_state 列表
        service_state = list(itertools.chain.from_iterable(service_state))
        # service_state = [service_state[0]] + service_state[1] + service_state[2] + service_state[3] + service_state[4]
        service_state = np.array(service_state)
        return observation_array, service_state

    def cal_link_SNR(self):
        link_SNR = np.zeros(80, dtype=float)
        for j in range(80):
            allocation, _, path_GSNR = self.only_check_action(j)
            if allocation:
                link_SNR[j] = np.min(path_GSNR, axis=0)[j]
        return link_SNR

    def check_action(self, action):
        tmp_topology = copy.deepcopy(self.topology)
        release_service(self.topology, self.service, self.service_dict)

        allocation = True
        outer_break = False
        Power = np.zeros((len(self.service.path), 80), dtype=float)
        path_GSNR = np.zeros((len(self.service.path), 80), dtype=float)

        for i in range((len(self.service.path) - 1)):
            if outer_break:
                break
            u = self.service.path[i]
            v = self.service.path[i + 1]
            # 检查波长是否空闲
            if not (self.topology[u][v]['wavelength_power'][action] == 0
                    or (np.isnan(self.topology[u][v]['wavelength_power'][action]))):
                # 找出所有值为零的元素的索引
                zero_indices = np.where(np.array(self.topology[u][v]['wavelength_power']) == 0.0)[0]
                # if len(zero_indices) > 0:
                #     action = np.random.choice(zero_indices)
                # else:
                allocation = False
                reason = "wavelength occupied!"
                break
            else:
                # 检查SNR是否满足要求
                # 获取链路参数
                distance = self.topology[u][v]['length']
                channels = 80
                if i == 0:
                    Power[i] = copy.deepcopy(self.topology[u][v]['wavelength_power'])
                    Power[i][action] = self.service.power

                frequencies = np.concatenate(
                    [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

                tmp = copy.deepcopy(self.topology[u][v]['wavelength_power'])
                tmp[action] = self.service.power
                tmp = np.array(tmp)

                if i != 0:
                    Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                                range(len(Power[i]))]
                    tmp = Power[i]
                    tmp = np.array(tmp)

                Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                path_GSNR[i] = GSNR

                if self.service.snr_requirement > GSNR[action]:
                    allocation = False
                    reason = "GSNR not satisfied!!"
                    break
                else:
                    Power[i + 1] = Power_after_transmission
                    #
                    # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                    for m in range(80):
                        if m != action and self.topology[u][v]['wavelength_service'][m] != 0:
                            tmp_service = self.service_dict.get(self.topology[u][v]['wavelength_service'][m], None)

                            if tmp_service.snr_requirement >= GSNR[m]:
                                allocation = False
                                reason = 'interference!'
                                outer_break = True
                                break
        self.topology = tmp_topology
        self.service_dict[self.service.service_id] = self.service

        return allocation, Power, path_GSNR

    def only_check_action(self, action):
        tmp_topology = copy.deepcopy(self.topology)
        tmp_servicedict = copy.deepcopy(self.service_dict)
        release_service(tmp_topology, self.service, tmp_servicedict)

        allocation = True
        outer_break = False

        Power = np.zeros((len(self.service.path), 80), dtype=float)
        path_GSNR = np.zeros((len(self.service.path)-1, 80), dtype=float)

        for i in range((len(self.service.path) - 1)):
            if outer_break:
                break
            u = self.service.path[i]
            v = self.service.path[i + 1]
            # 检查波长是否空闲
            if not (tmp_topology[u][v]['wavelength_power'][action] == 0
                    or (np.isnan(tmp_topology[u][v]['wavelength_power'][action]))):
                allocation = False
                reason = "wavelength occupied!"
                break
            else:
                # 检查SNR是否满足要求
                # 获取链路参数
                distance = tmp_topology[u][v]['length']
                channels = 80
                if i == 0:
                    Power[i] = copy.deepcopy(tmp_topology[u][v]['wavelength_power'])
                    Power[i][action] = self.service.power

                frequencies = np.concatenate(
                    [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

                tmp = copy.deepcopy(tmp_topology[u][v]['wavelength_power'])
                tmp[action] = self.service.power
                tmp = np.array(tmp)

                if i != 0:
                    Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                                range(len(Power[i]))]
                    tmp = Power[i]
                    tmp = np.array(tmp)

                Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                path_GSNR[i] = GSNR

                if self.service.snr_requirement > GSNR[action]:
                    allocation = False
                    reason = "GSNR not satisfied!!"
                    break
                else:
                    Power[i + 1] = Power_after_transmission

                    # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                    for m in range(80):
                        if m != action and tmp_topology[u][v]['wavelength_service'][m] != 0:
                            tmp_service = tmp_servicedict.get(tmp_topology[u][v]['wavelength_service'][m], None)
                            if tmp_service.snr_requirement >= GSNR[m]:
                                allocation = False
                                reason = 'interference!'
                                outer_break = True
                                break

        # Power = np.zeros((len(self.service.path), 80), dtype=float)
        # path_GSNR = np.zeros((len(self.service.path), 80), dtype=float)
        #
        # for i in range((len(self.service.path) - 1)):
        #     if outer_break:
        #         break
        #     u = self.service.path[i]
        #     v = self.service.path[i + 1]
        #     # 检查波长是否空闲
        #     if not (self.topology[u][v]['wavelength_power'][action] == 0
        #             or (np.isnan(self.topology[u][v]['wavelength_power'][action]))):
        #         # 找出所有值为零的元素的索引
        #         zero_indices = np.where(np.array(self.topology[u][v]['wavelength_power']) == 0.0)[0]
        #         # if len(zero_indices) > 0:
        #         #     action = np.random.choice(zero_indices)
        #         # else:
        #         allocation = False
        #         reason = "wavelength occupied!"
        #         break
        #     else:
        #         # 检查SNR是否满足要求
        #         # 获取链路参数
        #         distance = self.topology[u][v]['length']
        #         channels = 80
        #         if i == 0:
        #             Power[i] = copy.deepcopy(self.topology[u][v]['wavelength_power'])
        #             Power[i][action] = self.service.power
        #
        #         frequencies = np.concatenate(
        #             [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
        #
        #         tmp = copy.deepcopy(self.topology[u][v]['wavelength_power'])
        #         tmp[action] = self.service.power
        #
        #         if i != 0:
        #             Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
        #                         range(len(Power[i]))]
        #             tmp = Power[i]
        #
        #         Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
        #         path_GSNR[i] = GSNR
        #
        #         if self.service.snr_requirement > GSNR[action]:
        #             allocation = False
        #             reason = "GSNR not satisfied!!"
        #             break
        #         else:
        #             Power[i + 1] = Power_after_transmission
        #             #
        #             # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
        #             for m in range(80):
        #                 if m != action and self.topology[u][v]['wavelength_service'][m] != 0:
        #                     tmp_service = self.service_dict.get(self.topology[u][v]['wavelength_service'][m], None)
        #                     if tmp_service.snr_requirement >= GSNR[m]:
        #                         allocation = False
        #                         reason = 'interference!'
        #                         outer_break = True
        #                         break
        return allocation, Power, path_GSNR

    def only_calculate_reward(self, action, Power, path_GSNR):
        tmp_topology = copy.deepcopy(self.topology)
        tmp_frag = self.only_calculate_network_fragmentation(tmp_topology)
        tmp_servicedict = copy.deepcopy(self.service_dict)
        release_service(tmp_topology, self.service, tmp_servicedict)
        tmp_servicedict[self.service.service_id] = self.service
        for i in range(len(self.service.path) - 1):
            u = self.service.path[i]
            v = self.service.path[i + 1]
            tmp_topology[u][v]['wavelength_power'] = Power[i]
            tmp_topology[u][v]['wavelength_SNR'] = path_GSNR[i]
            capacity = 800 if tmp_topology[u][v]['wavelength_SNR'][action] > 26.5 else 400
            tmp_topology[u][v]['wavelength_utilization'][action] = self.service.bit_rate / capacity
            tmp_topology[u][v]['wavelength_service'][action] = self.service.service_id

            # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
            for wave in range(80):
                if wave != action and tmp_topology[u][v]['wavelength_service'][wave] != 0:
                    service_id = tmp_topology[u][v]['wavelength_service'][wave]
                    tmp_service = tmp_servicedict.get(service_id, None)
                    if tmp_service != None:
                        capacity = 800 if tmp_topology[u][v]['wavelength_SNR'][wave] > 26.5 else 400
                        # if tmp_service.bit_rate / capacity < 1:
                        tmp_topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity

        current_frag = self.only_calculate_network_fragmentation(tmp_topology)
        # reward = tmp_frag - current_frag
        reward = current_frag - tmp_frag

        return reward

    def remain_state(self):
        info = {}
        self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        self.current_service += 1
        if self.current_service == len(self.service_ids):
            self.episode_over = True
        else:
            self.episode_over = False
        reward = -0.01
        return self.get_observation(), reward, self.episode_over, self.service.utilization

    def DT_remain_state(self):
        info = {}
        self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        self.current_service += 1
        if self.current_service == len(self.service_ids):
            self.episode_over = True
        else:
            self.episode_over = False
        reward = -100
        return self.get_observation(), reward, self.episode_over, self.service.utilization

    def make_step(self, action):
        tmp_frag = self.only_calculate_network_fragmentation(self.topology)
        # print('service_id:', self.current_service, self.service.service_id)
        print('before_uti:', self.service.utilization)
        release_service(self.topology, self.service, self.service_dict)

        self.episode_over = False

        allocation = True
        outer_break = False
        reason = None
        zero_indices = []
        Power = np.zeros((len(self.service.path), 80), dtype=float)
        path_GSNR = np.zeros((len(self.service.path), 80), dtype=float)

        for i in range((len(self.service.path) - 1)):
            if outer_break:
                break
            u = self.service.path[i]
            v = self.service.path[i + 1]
            # 检查波长是否空闲
            if not (self.topology[u][v]['wavelength_power'][action] == 0
                    or (np.isnan(self.topology[u][v]['wavelength_power'][action]))):
                # 找出所有值为零的元素的索引
                zero_indices = np.where(np.array(self.topology[u][v]['wavelength_power']) == 0.0)[0]
                # if len(zero_indices) > 0:
                #     action = np.random.choice(zero_indices)
                # else:
                allocation = False
                reason = "wavelength occupied!"
                break
            else:
                # 检查SNR是否满足要求
                # 获取链路参数
                distance = self.topology[u][v]['length']
                channels = 80
                if i == 0:
                    Power[i] = copy.deepcopy(self.topology[u][v]['wavelength_power'])
                    Power[i][action] = self.service.power

                frequencies = np.concatenate(
                    [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

                tmp = copy.deepcopy(self.topology[u][v]['wavelength_power'])
                tmp[action] = self.service.power
                tmp = np.array(tmp)

                if i != 0:
                    Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                                range(len(Power[i]))]
                    tmp = Power[i]
                    tmp = np.array(tmp)

                Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                path_GSNR[i] = GSNR

                if self.service.snr_requirement > GSNR[action]:
                    allocation = False
                    reason = "GSNR not satisfied!!"
                    break
                else:
                    Power[i + 1] = Power_after_transmission

                    # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                    for m in range(80):
                        if m != action and self.topology[u][v]['wavelength_service'][m] != 0:
                            tmp_service = self.service_dict.get(self.topology[u][v]['wavelength_service'][m], None)

                            if tmp_service.snr_requirement >= GSNR[m]:
                                allocation = False
                                reason = 'interference!'
                                outer_break = True
                                break

        if allocation:
            total_utilization = 0
            self.service.wavelength = action
            self.service_dict[self.service.service_id] = self.service
            for i in range(len(self.service.path) - 1):
                u = self.service.path[i]
                v = self.service.path[i + 1]
                self.topology[u][v]['wavelength_power'] = Power[i]
                self.topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                capacity = 800 if self.topology[u][v]['wavelength_SNR'][action] > 26.5 else 400
                self.topology[u][v]['wavelength_utilization'][action] = self.service.bit_rate / capacity
                total_utilization += self.service.bit_rate / capacity
                self.topology[u][v]['wavelength_service'][action] = self.service.service_id

                # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                for wave in range(80):
                    if wave != action and self.topology[u][v]['wavelength_service'][wave] != 0:
                        service_id = self.topology[u][v]['wavelength_service'][wave]
                        tmp_service = self.service_dict.get(service_id, None)
                        if tmp_service != None:
                            capacity = 800 if self.topology[u][v]['wavelength_SNR'][wave] > 26.5 else 400
                            # if tmp_service.bit_rate / capacity < 1:
                            self.topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity

            self.service.utilization = total_utilization / len(self.service.path)

        # if not allocation:
        #     print("reason:", reason)
        #     reward = -100
        #     print('reward:', reward)
        #     self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        #     self.current_service += 1
        #     if self.current_service == len(self.service_ids):
        #         self.episode_over = True
        #
        #     return self.get_observation(), reward, self.episode_over, [allocation, zero_indices]

        print('after_uti:', self.service.utilization)
        self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        self.current_service += 1
        if self.current_service == len(self.service_ids):
            self.episode_over = True

        current_frag = self.only_calculate_network_fragmentation(self.topology)
        reward = tmp_frag - current_frag
        # reward = tmp_frag - current_frag
        # print('reward:', reward, current_frag, tmp_frag)
        return self.get_observation(), reward, self.episode_over, self.service.utilization

    def DT_make_step(self, action):
        tmp_frag = self._calculate_network_fragmentation()
        print('before_uti:', self.service.utilization)
        # print('service_id:', self.current_service, self.service.service_id)
        release_service(self.topology, self.service, self.service_dict)

        self.episode_over = False

        allocation = True
        outer_break = False
        reason = None
        zero_indices = []
        Power = np.zeros((len(self.service.path), 80), dtype=float)
        path_GSNR = np.zeros((len(self.service.path), 80), dtype=float)

        for i in range((len(self.service.path) - 1)):
            if outer_break:
                break
            u = self.service.path[i]
            v = self.service.path[i + 1]
            # 检查波长是否空闲
            if not (self.topology[u][v]['wavelength_power'][action] == 0
                    or (np.isnan(self.topology[u][v]['wavelength_power'][action]))):
                # 找出所有值为零的元素的索引
                zero_indices = np.where(np.array(self.topology[u][v]['wavelength_power']) == 0.0)[0]
                # if len(zero_indices) > 0:
                #     action = np.random.choice(zero_indices)
                # else:
                allocation = False
                reason = "wavelength occupied!"
                break
            else:
                # 检查SNR是否满足要求
                # 获取链路参数
                distance = self.topology[u][v]['length']
                channels = 80
                if i == 0:
                    Power[i] = copy.deepcopy(self.topology[u][v]['wavelength_power'])
                    Power[i][action] = self.service.power

                frequencies = np.concatenate(
                    [np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])

                tmp = copy.deepcopy(self.topology[u][v]['wavelength_power'])
                tmp[action] = self.service.power
                tmp = np.array(tmp)

                if i != 0:
                    Power[i] = [Power[i][j] if Power[i][j] != 0 and tmp[j] != 0 else tmp[j] for j in
                                range(len(Power[i]))]
                    tmp = Power[i]
                    tmp = np.array(tmp)

                Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                path_GSNR[i] = GSNR

                if self.service.snr_requirement > GSNR[action]:
                    allocation = False
                    reason = "GSNR not satisfied!!"
                    break
                else:
                    Power[i + 1] = Power_after_transmission

                    # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                    for m in range(80):
                        if m != action and self.topology[u][v]['wavelength_service'][m] != 0:
                            tmp_service = self.service_dict.get(self.topology[u][v]['wavelength_service'][m], None)

                            if tmp_service.snr_requirement >= GSNR[m]:
                                allocation = False
                                reason = 'interference!'
                                outer_break = True
                                break

        if allocation:
            total_utilization = 0
            self.service.wavelength = action
            self.service_dict[self.service.service_id] = self.service
            for i in range(len(self.service.path) - 1):
                u = self.service.path[i]
                v = self.service.path[i + 1]
                self.topology[u][v]['wavelength_power'] = Power[i]
                self.topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                capacity = 800 if self.topology[u][v]['wavelength_SNR'][action] > 26.5 else 400
                self.topology[u][v]['wavelength_utilization'][action] = self.service.bit_rate / capacity
                total_utilization += self.service.bit_rate / capacity
                self.topology[u][v]['wavelength_service'][action] = self.service.service_id

                # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                for wave in range(80):
                    if wave != action and self.topology[u][v]['wavelength_service'][wave] != 0:
                        service_id = self.topology[u][v]['wavelength_service'][wave]
                        tmp_service = self.service_dict.get(service_id, None)
                        if tmp_service != None:
                            capacity = 800 if self.topology[u][v]['wavelength_SNR'][wave] > 26.5 else 400
                            # if tmp_service.bit_rate / capacity < 1:
                            self.topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity

            self.service.utilization = total_utilization/len(self.service.path)
        # if not allocation:
        #     print("reason:", reason)
        #     reward = -100
        #     print('reward:', reward)
        #     self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        #     self.current_service += 1
        #     if self.current_service == len(self.service_ids):
        #         self.episode_over = True
        #
        #     return self.get_observation(), reward, self.episode_over, [allocation, zero_indices]

        print('after_uti:', self.service.utilization)
        self.service = self.service_to_be_sorting[self.service_ids[self.current_service]]
        self.current_service += 1
        if self.current_service == len(self.service_ids):
            self.episode_over = True
        current_frag = self._calculate_network_fragmentation()
        reward = -(tmp_frag - current_frag)
        # reward = tmp_frag - current_frag
        # print('reward:', reward, current_frag, tmp_frag)
        return self.get_observation(), reward, self.episode_over, self.service.utilization

    def get_reward(self, allocation):
        if not allocation:
            return -100
        else:
            # 计算整个网络的带宽碎片度
            all_utilization_rates = []
            utilizations = []
            for u, v in self.topology.edges():
                edge_data = self.topology[u][v]
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

            # 一致性因子：网络中带宽利用率的标准差
            utilizations_no_zeros = np.where(utilizations == 0, np.nan, utilizations)
            # 计算每个波长的平均利用率，忽略np.nan值
            average_utilization_per_wavelength = np.nanmean(utilizations_no_zeros, axis=0)
            average_utilization_per_wavelength_no_zeros = np.where(average_utilization_per_wavelength == 0, np.nan,
                                                                   average_utilization_per_wavelength)
            std_deviation = np.nanstd(average_utilization_per_wavelength_no_zeros)

            # 综合评估：带宽碎片度定义为利用率因子和一致性因子的加权和
            network_fragmentation = 0.9 * (1 - avg_utilization) + 0.1 * std_deviation  # 简化的计算公式，权重可以调整

            return self.map_value((1 - network_fragmentation)*10) * 100

    def _calculate_network_fragmentation(self):
        # 计算整个网络的带宽碎片度
        all_utilization_rates = []
        utilizations = []
        for u, v in self.topology.edges():
            edge_data = self.topology[u][v]
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

        # 一致性因子：网络中带宽利用率的标准差
        utilizations_no_zeros = np.where(utilizations == 0, np.nan, utilizations)
        # 计算每个波长的平均利用率，忽略np.nan值
        average_utilization_per_wavelength = np.nanmean(utilizations_no_zeros, axis=0)
        average_utilization_per_wavelength_no_zeros = np.where(average_utilization_per_wavelength == 0, np.nan,
                                                               average_utilization_per_wavelength)
        std_deviation = np.nanstd(average_utilization_per_wavelength_no_zeros)

        # 综合评估：带宽碎片度定义为利用率因子和一致性因子的加权和
        network_fragmentation = 0.9 * (1 - avg_utilization) + 0.1 * std_deviation  # 简化的计算公式，权重可以调整

        # return network_fragmentation
        return self.map_value((1 - network_fragmentation) * 10) * 10

    def only_calculate_network_fragmentation(self, tmp_topology):
        # 计算整个网络的带宽碎片度
        all_utilization_rates = []
        utilizations = []
        for u, v in tmp_topology.edges():
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

        return network_fragmentation
        # return self.map_value((1 - network_fragmentation) * 10) * 100

    def _calculate_network_utilization(self):
        # 计算整个网络的带宽碎片度
        all_utilization_rates = []
        utilizations = []
        for u, v in self.topology.edges():
            edge_data = self.topology[u][v]
            utilizations.append(edge_data['wavelength_utilization'])
            occupied_wavelengths = np.count_nonzero(np.array(edge_data['wavelength_power']) > 0)
            total_utilization = 0
            for i in range(len(edge_data['wavelength_power'])):
                total_utilization += edge_data['wavelength_utilization'][i]
            if occupied_wavelengths == 0:
                utilization_rate = 0
            else:
                utilization_rate = total_utilization / occupied_wavelengths
            all_utilization_rates.append(utilization_rate)

        # 利用率因子：网络中所有链路的带宽利用率的平均值
        avg_utilization = np.mean(all_utilization_rates)
        return avg_utilization

    def map_value(self, value):
        if value < 4:
            return 1
        elif value > 7:
            return 10
        else:
            # 等比例映射到1-10之间
            return 3 * value - 11
