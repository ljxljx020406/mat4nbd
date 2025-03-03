from multiband_optical_network_env import MultibandOpticalNetworkEnv
import pickle
import numpy as np
import random

def evaluate_and_sort_actions(env, wave1):
    action_rewards = {}
    # print('service:', env.service.service_id)
    for wave in range(80):
        allocation, Power, path_GSNR = env.only_check_action(wave)
        if not allocation:
            reward = -100
        else:
            reward = env.only_calculate_reward(wave, Power, path_GSNR)
        if wave == wave1:
            reward = 0

        action_rewards[wave] = reward
        # print(wave, reward)

    # 将字典按照奖励值从大到小排序
    sorted_action_rewards = sorted(action_rewards.items(), key=lambda item: item[1], reverse=True)

    return sorted_action_rewards[:2]

topology_file= '../topology/new_usnet_500service_with_fragmentation5-2.pkl'
trajectories = []

for i in range(200):
    print('i=', i)
    traj = {
        'link_state': [],
        'service_state': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    done = False
    with open('../service/new_usnet_500services_after_release5-2.pkl', 'rb') as f:
        service_dict = pickle.load(f)
    with open('../service/new_usnet_500services_to_be_sorting5-2.pkl', 'rb') as f:
        service_to_be_sorting = pickle.load(f)

    env = MultibandOpticalNetworkEnv(topology_file=topology_file, service_dict=service_dict,
                                              service_to_be_sorting=service_to_be_sorting)

    initial_util = env._calculate_network_utilization()

    (link_state0, service_state0) = env.reset(topology_file, service_dict, service_to_be_sorting)
    # print('observation0:', len(observation0))
    traj['link_state'].append(link_state0)
    traj['service_state'].append(service_state0)
    while not done:
        top_five_actions = evaluate_and_sort_actions(env, env.service.wavelength)
        # print('origin_wave:', env.service.wavelength)
        print('top5action:', top_five_actions)
        action, tmp_reward = random.choice(top_five_actions)
        print('action:', action, tmp_reward)
        if tmp_reward != -100:
            (link_state, service_state), reward, done, info = env.make_step(action)
        else:
            print('allocation failure!!!!!')
            (link_state, service_state), reward, done, info = env.remain_state()
        # print('reward:', reward)
        traj['actions'].append(action)
        # print('action:', action)
        traj['rewards'].append(reward)
        # print('reward:', reward)
        traj['dones'].append(done)
        # print('done:', done)
        traj['link_state'].append(link_state)
        traj['service_state'].append(service_state)
        # print('service_State:', service_state)

    cur_util = env._calculate_network_utilization()
    print('delta_util:', cur_util-initial_util, cur_util, initial_util)
    # print(traj['rewards'])
    traj['link_state'].pop()
    traj['service_state'].pop()
    trajectories.append(traj)
    print('traj_rewards:', traj['rewards'])
    # print('traj:', traj['service_state'])
    # print('traj:', traj['actions'], len(traj['actions']), traj['rewards'], len(traj['rewards']), traj['dones'], len(traj['dones']))
    # print('link_state:', len(traj['link_state']), len(traj['link_state'][0]), len(traj['link_state'][0][0]), len(traj['service_state']))
# print(trajectories)

with open('../data/new_200good_data5-2-1-1.pkl', 'wb') as f:
    pickle.dump(trajectories, f)