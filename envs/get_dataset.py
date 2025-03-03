import sys
import os

from multiband_optical_network_env import MultibandOpticalNetworkEnv
import pickle
import numpy as np

'''
数据集格式：
trajectories中包含很多traj，每个traj是一个字典，包含['observations']、['actions']、['rewards']、['dones']
observation: 业务所经过链路、80个波长处的GSNR + 业务比特率 + ... 共1*3361
action: [0, 79]中无业务的载波处选1
reward: 分配失败-100，分配成功(1-带宽碎片度)*10
done: 已遍历所有待整理业务或分配失败
'''

topology_file= '../topology/usnet_500service_with_fragmentation2.pkl'
trajectories = []

for i in range(10000):
    print('i=', i)
    traj = {
        'link_state': [],
        'service_state': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    done = False
    with open('../service/usnet_500services_after_release2.pkl', 'rb') as f:
        service_dict = pickle.load(f)
    with open('../service/usnet_500services_to_be_sorting2.pkl', 'rb') as f:
        service_to_be_sorting = pickle.load(f)

    env = MultibandOpticalNetworkEnv(topology_file=topology_file, service_dict=service_dict,
                                              service_to_be_sorting=service_to_be_sorting)
    initial_util = env._calculate_network_utilization()

    link_state0, service_state0 = env.reset(topology_file, service_dict, service_to_be_sorting)
    # print('observation0:', len(observation0))
    traj['link_state'].append(link_state0)
    traj['service_state'].append(service_state0)
    # while not done:
    #     max_reward = - np.inf
    #     max_action = 81
    #     for _ in range(5):
    #         action = env.action_space.sample()
    #         allocation, Power, path_GSNR = env.only_check_action(action)
    #         tmp_reward = env.only_calculate_reward(action, Power, path_GSNR)
    #         if tmp_reward > max_reward:
    #             max_action = action
    #     allocation, Power, path_GSNR = env.only_check_action(max_action)
    #     if not allocation:
    #         (link_state, service_state), reward, done, info = env.remain_state()
    #     else:
    #         (link_state, service_state), reward, done, info = env.make_step(max_action)
    #     # print('fragmentation:', env._calculate_network_fragmentation())
    #
    #     traj['actions'].append(max_action)
    while not done:
        action = env.action_space.sample()
        allocation, Power, path_GSNR = env.only_check_action(action)
        if not allocation:
            (link_state, service_state), reward, done, info = env.remain_state()
        else:
            (link_state, service_state), reward, done, info = env.make_step(action)
        # print('action:', action)
        traj['rewards'].append(reward)
        traj['actions'].append(action)
        # print('reward:', reward)
        traj['dones'].append(done)
        # print('done:', done)
        traj['link_state'].append(link_state)
        traj['service_state'].append(service_state)
        # print('service_State:', service_state)

    cur_util = env._calculate_network_utilization()
    print('delta_util:', cur_util - initial_util, cur_util, initial_util)
    # print(traj['rewards'])
    traj['link_state'].pop()
    traj['service_state'].pop()
    trajectories.append(traj)
    print('traj:', traj['service_state'])
    print('action:', max(traj['actions']))
    # print('traj:', traj['actions'], len(traj['actions']), traj['rewards'], len(traj['rewards']), traj['dones'], len(traj['dones']))
    # print('link_state:', len(traj['link_state']), len(traj['link_state'][0]), len(traj['link_state'][0][0]), len(traj['service_state']), len(traj['rewards']))
# print(trajectories)

with open('../data/10000random_data2-2.pkl', 'wb') as f:
    pickle.dump(trajectories, f)