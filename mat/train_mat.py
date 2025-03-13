import sys
import os
import numpy as np
import torch
import random
import pickle
import time
import pandas as pd

# sys.path.append("../")
# sys.path.append(os.path.abspath('../envs'))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

envs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../envs"))
sys.path.append(envs_path)

from mat.config import get_config
from mat.runner.shared.football_runner import FootballRunner as Runner
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from envs.multiband_optical_network_env import MultibandOpticalNetworkEnv
from background.main_functions import new_service, naive_RWA, release_service

rng = random.Random(42)


def train_mat(topology_file, model_path=None):

    with open(topology_file, 'rb') as f:
        topology = pickle.load(f)

    # 定义仿真参数
    lambda_rate = 1  # 到达率
    erlang_values = [20, 50, 80, 110, 140, 170, 200]  # Erlang 目标值
    # 仿真时间
    simulation_time1 = 200
    stage_duration = simulation_time1 / len(erlang_values)  # 每个阶段的时间
    # mu_rate = 1 / 140  # 持续时间的倒数，平均持续时间为180秒

    # 初始化变量
    time = 0
    calls_in_progress = 0
    total_calls = 0
    call_arrivals = []
    service_dict = {}
    # 开始仿真
    while time < simulation_time1:
        stage_index = min(int(time / stage_duration), len(erlang_values) - 1)
        target_erlang = erlang_values[stage_index]
        mu_rate = lambda_rate / target_erlang

        # 下一次到达时间（到达间隔时间服从参数为λ的指数分布）
        time_to_next_arrival = np.random.exponential(1 / lambda_rate)
        time += time_to_next_arrival
        print(f'Time: {time:.2f}, Stage: {stage_index + 1}, Target Erlangs: {target_erlang}, New mu_rate: {mu_rate:.6f}')

        if time >= simulation_time1:
            break

        # 记录呼叫到达时间
        call_arrivals.append(time)

        # 计算呼叫持续时间（服从参数为μ的指数分布）
        call_duration = np.random.exponential(1 / mu_rate)
        call_departure_time = time + call_duration

        # 更新正在进行的呼叫数量
        calls_in_progress += 1

        tmp_service = new_service(topology, total_calls, 100, 700)
        total_calls += 1
        tmp_service.arrival_time = time
        tmp_service.holding_time = call_departure_time
        service_dict[tmp_service.service_id] = tmp_service

        for i in service_dict.keys():
            if service_dict[i].holding_time <= time:
                release_service(topology, service_dict[i], service_dict)