import copy
import sys
import os
import random
import pickle
import numpy as np
sys.path.append(os.path.abspath('../envs'))
sys.path.append(os.path.abspath('../data'))
sys.path.append(os.path.abspath('../topology'))
sys.path.append(os.path.abspath('../service'))
sys.path.append(os.path.abspath('../background'))
from main_functions import release_service, one_link_transmission, first_fit, _get_node_pair, naive_RWA
from utils import Service

rng = random.Random(46)


def new_service(topology, services_processed_since_reset):
    '''
    生成一个随机业务
    '''
    src, src_id, dst, dst_id = _get_node_pair(topology)

    # list of possible bit-rates for the request
    # 定义可能的值和概率
    # values = [800, 400]
    # probabilities = [0.25, 0.75]
    #
    # # 随机选择
    # bit_rate = np.random.choice(values, p=probabilities)
    # bit_rate = 400

    bit_rate = rng.randint(100, 750)

    service = Service(service_id=services_processed_since_reset, source=src, source_id=src_id,
                      destination=dst, destination_id=dst_id,
                      bit_rate=bit_rate)

    # services_processed_since_reset += 1

    return service


with open('../topology/usnet_topology_1path.h5', 'rb') as f:
    topology = pickle.load(f)

# 定义仿真参数
lambda_rate = 1  # 到达率
mu_rate = 1 / 200  # 持续时间的倒数，平均持续时间为180秒

# 计算目标Erlang值
target_erlangs = lambda_rate / mu_rate
print(target_erlangs)

# 仿真时间
simulation_time1 = 10000

# 初始化变量
time = 0
calls_in_progress = 0
total_calls = 0
call_arrivals = []
call_departures = []
service_dict = {}
# 开始仿真
while time < simulation_time1:
    # 下一次到达时间（到达间隔时间服从参数为λ的指数分布）
    time_to_next_arrival = np.random.exponential(1 / lambda_rate)
    time += time_to_next_arrival
    print('time:',time)

    if time >= simulation_time1:
        break

    # 记录呼叫到达时间
    call_arrivals.append(time)

    # 计算呼叫持续时间（服从参数为μ的指数分布）
    call_duration = np.random.exponential(1 / mu_rate)
    call_departure_time = time + call_duration

    # 更新正在进行的呼叫数量
    calls_in_progress += 1

    tmp_service = new_service(topology, total_calls)
    total_calls += 1
    print('id/src/dst/bitrate:', tmp_service.service_id, tmp_service.source_id, tmp_service.destination_id, tmp_service.bit_rate)
    tmp_service.arrival_time = time
    tmp_service.holding_time = call_departure_time
    service_dict[tmp_service.service_id] = tmp_service

# simulation_time2 = 700
# while time < simulation_time2:
#     # 下一次到达时间（到达间隔时间服从参数为λ的指数分布）
#     time_to_next_arrival = np.random.exponential(1 / lambda_rate)
#     time += time_to_next_arrival
#     print('time:',time)
#
#     if time >= simulation_time2:
#         break
#
#     # 记录呼叫到达时间
#     call_arrivals.append(time)
#
#     # 计算呼叫持续时间（服从参数为μ的指数分布）
#     call_duration = np.random.exponential(1 / mu_rate)
#     call_departure_time = time + call_duration
#
#     # 更新正在进行的呼叫数量
#     calls_in_progress += 1
#
#     tmp_service = new_large_service(topology, total_calls)
#     total_calls += 1
#     print('id/src/dst/bitrate:', tmp_service.service_id, tmp_service.source_id, tmp_service.destination_id, tmp_service.bit_rate)
#     tmp_service.arrival_time = time
#     tmp_service.holding_time = call_departure_time
#     service_dict[tmp_service.service_id] = tmp_service

with open('../service/usnet_200erl_1w.pkl', 'wb') as f:
    pickle.dump(service_dict, f)