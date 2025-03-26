import copy
import sys
import os
import numpy as np
import torch
import random
import pickle
import setproctitle
import wandb
import socket
from pathlib import Path
from tqdm import tqdm
import time
import pandas as pd

sys.path.append("../")
# sys.path.append(os.path.abspath('../env'))
sys.path.append("../../env")
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)
#
# envs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../env"))
# sys.path.append(envs_path)

from mat.config import get_config
from mat.runner.shared.optinet_runner import OptinetRunner as Runner
from mat.utils.shared_buffer import SharedReplayBuffer
from envs.multiband_optical_network_env import MultibandOpticalNetworkEnv
from background.main_functions import new_service, naive_RWA, release_service, select_sorting_services

rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)

num_agent = 10


def parse_args(args, parser):
    parser.add_argument('--n_agent', type=int, default=20)
    all_args = parser.parse_known_args(args)[0]

    return all_args

def episode_train(args, buffer, episode, episodes, topology, service_dict, service_to_be_sorting, num_agents, blocked_service):
    env = MultibandOpticalNetworkEnv(topology, service_dict, service_to_be_sorting, num_agents, blocked_service)
    # state = env.reset(topology, service_dict, service_to_be_sorting)
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        # print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif all_args.device == 'mps' and torch.backends.mps.is_available():
        # print("choose to use MPS on MacBook...")
        device = torch.device('mps')
        torch.set_num_threads(all_args.n_training_threads)
    else:
        # print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    config = {
        "all_args": all_args,
        "env": env,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "buffer": buffer
    }
    runner = Runner(config)
    topology, service_dict, block_flag = runner.run(episode, episodes)

    env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

    return topology, service_dict, block_flag
def main(args):
    with open('../topology/usnet_topology_1path.h5', 'rb') as f:
        topology = pickle.load(f)
    parser = get_config()
    all_args = parse_args(args, parser)
    dummy_env = MultibandOpticalNetworkEnv(topology, [], {}, num_agent, [])
    buffer = SharedReplayBuffer(all_args, num_agent, dummy_env.observation_space[0], dummy_env.observation_space[0], dummy_env.action_space[0], all_args.env_name)


    # 定义仿真参数
    lambda_rate = 1  # 到达率
    erlang_values = [150, 200, 250, 300, 350, 400]  # Erlang 目标值

    # 初始化变量
    time = 0
    total_calls = 1
    service_dict = {}
    block_num = 0
    episode = 0
    episodes = 1000  # 总被阻塞数 —— num_env_steps/episode_length = 10e6/200 = 5000

    stage_duration = episodes / len(erlang_values)  # 每个阶段的时间

    # 创建 tqdm 进度条
    progress_bar = tqdm(total=episodes, desc="Training Progress", unit="blocks")

    # 开始仿真
    while episode < episodes:
        stage_index = min(int(block_num / stage_duration), len(erlang_values) - 1)
        target_erlang = erlang_values[stage_index]
        mu_rate = lambda_rate / target_erlang

        # 下一次到达时间（到达间隔时间服从参数为λ的指数分布）
        time_to_next_arrival = rng1.exponential(1 / lambda_rate)
        time += time_to_next_arrival
        # print(f'Time: {time:.2f}, Stage: {stage_index + 1}, Target Erlangs: {target_erlang}, New mu_rate: {mu_rate:.6f}')

        # 计算呼叫持续时间（服从参数为μ的指数分布）
        call_duration = rng2.exponential(1 / mu_rate)
        call_departure_time = time + call_duration

        tmp_service = new_service(topology, total_calls, 100, 650)
        # print('id/src/dst/bitrate:', tmp_service.service_id, tmp_service.source_id, tmp_service.destination_id,
        #       tmp_service.bit_rate)
        total_calls += 1
        tmp_service.arrival_time = time
        tmp_service.holding_time = call_departure_time
        # service_dict[tmp_service.service_id] = tmp_service

        static_dict = copy.deepcopy(service_dict)
        for i in static_dict.keys():
            if static_dict[i].holding_time <= time:
                release_service(topology, static_dict[i], service_dict)

        path, wavelength, info = naive_RWA(topology, tmp_service, service_dict)
        if path == None:
            episode += 1
            progress_bar.update(1)  # 更新进度条
            src = tmp_service.source_id
            dst = tmp_service.destination_id
            tmp_service.path = topology.graph['ksp'][str(src), str(dst)][0].node_list
            # print('path:', tmp_service.path)
            service_to_be_sorting = select_sorting_services(service_dict, tmp_service, time, num_agent, 0.4)
            if len(service_to_be_sorting) == 0:
                continue
            topology, service_dict, block_flag = episode_train(args, buffer, episode, episodes, topology, service_dict, service_to_be_sorting, num_agent, tmp_service)
            path, wavelength, info = naive_RWA(topology, tmp_service, service_dict)
            if wavelength == None:
                block_num += 1

    progress_bar.close()  # 训练结束，关闭进度条
    print('block:', block_num/total_calls)


if __name__ == "__main__":
    main(sys.argv[1:])