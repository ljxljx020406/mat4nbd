import random

import torch
import numpy as np
import torch.nn.functional as F
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space

seed = 42
np.random.seed(seed)

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, env_name):
        self.episode_length = args.episode_length
        self.max_buffer_episodes = args.max_buffer_episodes
        self.max_steps = self.max_buffer_episodes * (self.episode_length + 1)
        self.episode_start = 0
        self.batch_size = args.batch_size

        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.env_name = env_name

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        # 将数组的第一维由self.episode_length+1 改为 self.max_steps
        self.share_obs = np.zeros((self.max_steps, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.max_steps, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.max_steps, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.max_steps, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.max_steps, num_agents, 1), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.max_steps, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.max_steps, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.max_steps, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.max_steps, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.max_steps, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0
        self.episode_start = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        # 将self.step替换为idx
        idx = self.step % self.max_steps
        self.share_obs[idx + 1] = share_obs.copy()
        self.obs[idx + 1] = obs.copy()
        self.rnn_states[idx + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[idx + 1] = rnn_states_critic.copy()
        self.actions[idx] = actions.copy()
        self.action_log_probs[idx] = action_log_probs.copy()
        self.value_preds[idx] = value_preds.copy()
        self.rewards[idx] = rewards.copy()
        self.masks[idx + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[idx + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[idx + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[idx + 1] = available_actions.copy()

        # self.step = (self.step + 1) % self.episode_length
        if self.step % self.episode_length == 0:
            self.episode_start = self.step
        self.step += 1

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        # self.share_obs[0] = self.share_obs[-1].copy()
        # self.obs[0] = self.obs[-1].copy()
        # self.rnn_states[0] = self.rnn_states[-1].copy()
        # self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        # self.masks[0] = self.masks[-1].copy()
        # self.bad_masks[0] = self.bad_masks[-1].copy()
        # self.active_masks[0] = self.active_masks[-1].copy()
        # if self.available_actions is not None:
        #     self.available_actions[0] = self.available_actions[-1].copy()
        pass

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        # self.rnn_states[0] = self.rnn_states[-1].copy()
        # self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        # self.masks[0] = self.masks[-1].copy()
        # self.bad_masks[0] = self.bad_masks[-1].copy()
        pass

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        # self.value_preds[-1] = next_value
        # gae = 0
        # for step in reversed(range(self.rewards.shape[0])):
        #     if self._use_popart or self._use_valuenorm:
        #         delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
        #             self.value_preds[step + 1]) * self.masks[step + 1] \
        #                 - value_normalizer.denormalize(self.value_preds[step])
        #         # print('delta:', delta, step)
        #         # print(self.rewards[step])
        #         # print(self.gamma)
        #         gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
        #
        #         # here is a patch for mpe, whose last step is timeout instead of terminate
        #         if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
        #             gae = 0
        #
        #         self.advantages[step] = gae
        #         self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
        #     else:
        #         delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
        #                 self.masks[step + 1] - self.value_preds[step]
        #         gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
        #
        #         # here is a patch for mpe, whose last step is timeout instead of terminate
        #         if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
        #             gae = 0
        #
        #         self.advantages[step] = gae
        #         self.returns[step] = gae + self.value_preds[step]
        # print('compute_return:', self.advantages, self.advantages.mean(), self.advantages.std())

        last_step = self.step % self.max_steps
        self.value_preds[last_step] = next_value
        gae = 0
        # print('buffer_compute:', self.episode_start, self.step)
        for i in reversed(range(self.episode_start, self.step-1)):
            idx = i % self.max_steps
            idx_next = (i + 1) % self.max_steps

            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[idx] + self.gamma * value_normalizer.denormalize(self.value_preds[idx_next]) * \
                        self.masks[idx_next] \
                        - value_normalizer.denormalize(self.value_preds[idx])
                gae = delta + self.gamma * self.gae_lambda * self.masks[idx_next] * gae
                self.advantages[idx] = gae
                self.returns[idx] = gae + value_normalizer.denormalize(self.value_preds[idx])
            else:
                delta = self.rewards[idx] + self.gamma * self.value_preds[idx_next] * self.masks[idx_next] \
                        - self.value_preds[idx]
                gae = delta + self.gamma * self.gae_lambda * self.masks[idx_next] * gae
                self.advantages[idx] = gae
                self.returns[idx] = gae + self.value_preds[idx]
        print('compute_return:', self.advantages[self.episode_start:self.step].mean(), self.advantages[self.episode_start:self.step].std())

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into. 训练时，将数据划分为num_mini_batch份
        :param mini_batch_size: (int) number of samples in each minibatch. 每个mini_batch的样本数量
        """
        # episode_length, num_agents = self.rewards.shape[0:2]
        # batch_size = episode_length  # 原样本总数
        #
        # if mini_batch_size is None:
        #     # assert batch_size >= num_mini_batch, (
        #     #     "PPO requires the number of processes ({}) "
        #     #     "* number of steps ({}) = {} "
        #     #     "to be greater than or equal to the number of PPO mini batches ({})."
        #     #     "".format(n_rollout_threads, episode_length,
        #     #               n_rollout_threads * episode_length,
        #     #               num_mini_batch))
        #     mini_batch_size = batch_size // num_mini_batch
        #
        # rand = torch.randperm(batch_size).numpy()
        # sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        # rows, cols = _shuffle_agent_grid(batch_size, num_agents)
        #
        # # keep (num_agent, dim)
        # share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[1:])
        # share_obs = share_obs[rows, cols]
        # obs = self.obs[:-1].reshape(-1, *self.obs.shape[1:])
        # obs = obs[rows, cols]
        # rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[1:])
        # rnn_states = rnn_states[rows, cols]
        # rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[1:])
        # rnn_states_critic = rnn_states_critic[rows, cols]
        # actions = self.actions.reshape(-1, *self.actions.shape[1:])
        # actions = actions[rows, cols]
        # if self.available_actions is not None:
        #     available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[1:])
        #     available_actions = available_actions[rows, cols]
        # value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[1:])
        # value_preds = value_preds[rows, cols]
        # returns = self.returns[:-1].reshape(-1, *self.returns.shape[1:])
        # returns = returns[rows, cols]
        # masks = self.masks[:-1].reshape(-1, *self.masks.shape[1:])
        # masks = masks[rows, cols]
        # active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[1:])
        # active_masks = active_masks[rows, cols]
        # action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[1:])
        # action_log_probs = action_log_probs[rows, cols]
        # advantages = advantages.reshape(-1, *advantages.shape[1:])
        # advantages = advantages[rows, cols]
        #
        # for indices in sampler:
        #     # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
        #     share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[1:])
        #     obs_batch = obs[indices].reshape(-1, *obs.shape[1:])
        #     rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[1:])
        #     rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[1:])
        #     actions_batch = actions[indices].reshape(-1, *actions.shape[1:])
        #     if self.available_actions is not None:
        #         available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[1:])
        #     else:
        #         available_actions_batch = None
        #     value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[1:])
        #     return_batch = returns[indices].reshape(-1, *returns.shape[1:])
        #     masks_batch = masks[indices].reshape(-1, *masks.shape[1:])
        #     active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[1:])
        #     old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[1:])
        #     if advantages is None:
        #         adv_targ = None
        #     else:
        #         adv_targ = advantages[indices].reshape(-1, *advantages.shape[1:])
        #
        #     yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        #           value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        #           adv_targ, available_actions_batch

        num_steps = min(self.step, self.max_steps)  # 有效的 step 数
        sample_size = min(num_steps, self.batch_size)

        # 自动设置 mini_batch_size
        if mini_batch_size is None:
            assert num_mini_batch is not None, "Must provide mini_batch_size or num_mini_batch"
            mini_batch_size = sample_size // num_mini_batch

        sampled_indices = np.random.choice(num_steps, sample_size, replace=False)
        np.random.shuffle(sampled_indices)

        # 按 mini_batch_size 分段
        for start in range(0, sample_size, mini_batch_size):
            end = start + mini_batch_size
            indices = sampled_indices[start:end]  # 当前 batch 中的 step 索引

            # 收集该 mini-batch 对应 step 的数据
            share_obs_batch = self.share_obs[indices]  # (B, N, D)
            obs_batch = self.obs[indices]
            rnn_states_batch = self.rnn_states[indices]
            rnn_states_critic_batch = self.rnn_states_critic[indices]
            actions_batch = self.actions[indices]
            value_preds_batch = self.value_preds[indices]
            return_batch = self.returns[indices]
            masks_batch = self.masks[indices]
            active_masks_batch = self.active_masks[indices]
            old_action_log_probs_batch = self.action_log_probs[indices]
            adv_targ = advantages[indices]
            available_actions_batch = self.available_actions[indices] if self.available_actions is not None else None

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
