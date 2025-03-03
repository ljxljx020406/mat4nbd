from multiband_optical_network_env import MultibandOpticalNetworkEnv
import pickle
import numpy as np
import random

def evaluate_and_sort_actions(env, wave1):
    action_rewards = {}
    for wave in range(80):
        allocation, Power, path_GSNR = env.only_check_action(wave)
        # print('Power:', Power, path_GSNR)
        if not allocation:
            reward = -100
        else:
            reward = env.only_calculate_reward(wave, Power, path_GSNR)
        if wave == wave1:
            print(reward)
            if reward != 0:
                print('error!!!', env.service.path)

        action_rewards[wave] = reward

    # 将字典按照奖励值从大到小排序
    sorted_action_rewards = sorted(action_rewards.items(), key=lambda item: item[1], reverse=True)

    return sorted_action_rewards


random.seed(423)

for i in range(10):
    topology_file= '../topology/usnet_150services2.pkl'
    with open('../service/usnet_150service_dict2.pkl', 'rb') as f:
        service_dict = pickle.load(f)
    with open('../service/usnet_150services_test2.pkl', 'rb') as f:
        service_to_be_sorting = pickle.load(f)
        # 将键值对转换为列表
        items = list(service_to_be_sorting.items())

        # 打乱列表
        random.shuffle(items)

        # 如果你需要回到字典形式
        service_to_be_sorting_shuffled = dict(items)

    env = MultibandOpticalNetworkEnv(topology_file=topology_file, service_dict=service_dict,
                                     service_to_be_sorting=service_to_be_sorting_shuffled)

    observation = env.reset(topology_file, service_dict, service_to_be_sorting_shuffled)

    wave = env.service.wavelength
    print('service:', env.service.service_id, wave)
    top_ten_actions = evaluate_and_sort_actions(env, wave)
    print(top_ten_actions)