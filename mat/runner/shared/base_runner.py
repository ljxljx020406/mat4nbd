import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from mat.utils.shared_buffer import SharedReplayBuffer
from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['env']
        # self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V  # True：使用集中式价值函数（适用于多智能体环境）
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state  # 是否使用观测信息而非状态信息
        self.num_env_steps = self.all_args.num_env_steps  # 训练过程中执行的环境步骤总数
        self.episode_length = self.all_args.episode_length  # 智能体在一次回合中执行的最大步数
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 评估时并行运行的环境线程数
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay  # True：学习率在训练过程中按线性衰减
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb  # True：使用wandb进行日志记录、训练监控、可视化等
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval  # 保存模型的时间间隔（按训练步数）
        self.use_eval = self.all_args.use_eval  # True：在训练过程中进行评估
        self.eval_interval = self.all_args.eval_interval  # 评估的时间间隔
        self.log_interval = self.all_args.log_interval  # 记录日志的时间间隔

        # dir
        self.model_dir = self.all_args.model_dir  # 模型的存储目录

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # print("obs_space: ", self.envs.observation_space)
        # print("share_obs_space: ", self.envs.share_observation_space)
        # print("act_space: ", self.envs.action_space)

        # policy network
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             share_observation_space,
                             self.envs.action_space[0],
                             self.num_agents,
                             device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)
        if "buffer" in config:
            self.buffer = config["buffer"]
        else:
            # buffer
            self.buffer = SharedReplayBuffer(self.all_args,
                                            self.num_agents,
                                            self.envs.observation_space[0],
                                            share_observation_space,
                                            self.envs.action_space[0],
                                             self.all_args.env_name)

    def run(self, episode, episodes):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        # self.trainer.prep_rollout()
        # if self.buffer.available_actions is None:  # 连续动作空间
        #     next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
        #                                                  np.concatenate(self.buffer.obs[-1]),
        #                                                  np.concatenate(self.buffer.rnn_states_critic[-1]),
        #                                                  np.concatenate(self.buffer.masks[-1]))
        # else:  # 离散动作空间
        #     next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
        #                                                  np.concatenate(self.buffer.obs[-1]),
        #                                                  np.concatenate(self.buffer.rnn_states_critic[-1]),
        #                                                  np.concatenate(self.buffer.masks[-1]),
        #                                                  np.concatenate(self.buffer.available_actions[-1]))
        # next_values = np.array(_t2n(next_values))
        # self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

        self.trainer.prep_rollout()
        idx = self.buffer.step % self.buffer.max_steps
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(
                np.concatenate(self.buffer.share_obs[idx]),
                np.concatenate(self.buffer.obs[idx]),
                np.concatenate(self.buffer.rnn_states_critic[idx]),
                np.concatenate(self.buffer.masks[idx]))
        else:
            next_values = self.trainer.policy.get_values(
                np.concatenate(self.buffer.share_obs[idx]),
                np.concatenate(self.buffer.obs[idx]),
                np.concatenate(self.buffer.rnn_states_critic[idx]),
                np.concatenate(self.buffer.masks[idx]),
                np.concatenate(self.buffer.available_actions[idx]))
        next_values = np.array(_t2n(next_values))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.policy.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info. 记录训练过程中的更新信息
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info. 记录训练环境的信息
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
