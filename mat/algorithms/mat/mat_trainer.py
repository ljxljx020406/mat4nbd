import numpy as np
import torch
import torch.nn as nn
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check


class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param  # 决定了策略更新的幅度
        self.ppo_epoch = args.ppo_epoch  # PPO epoch数量，每次训练执行PPO更新的轮数
        self.num_mini_batch = args.num_mini_batch  # 每次训练时，数据被划分成的 mini-batch 数量
        self.data_chunk_length = args.data_chunk_length  # 每次从经验缓冲区（Replay Buffer）中获取数据的长度
        self.value_loss_coef = args.value_loss_coef  # 值函数损失的系数，决定了值函数损失在总损失中的比重
        self.entropy_coef = args.entropy_coef  # 熵项的系数，鼓励策略的多样性，防止过早收敛
        self.max_grad_norm = args.max_grad_norm  # 梯度裁剪的最大值，用于防止梯度爆炸
        self.huber_delta = args.huber_delta  # Huber 损失函数中的 delta 参数，控制损失的平滑度

        self._use_recurrent_policy = args.use_recurrent_policy  # 是否使用 循环神经网络（RNN） 政策，适用于部分观测问题
        self._use_naive_recurrent = args.use_naive_recurrent_policy  # 是否使用简单的递归策略，而不是基于 Transformer 等复杂结构
        self._use_max_grad_norm = args.use_max_grad_norm  # 是否启用最大梯度裁剪
        self._use_clipped_value_loss = args.use_clipped_value_loss  # 是否使用剪切的价值损失，帮助防止值函数的过度更新
        self._use_huber_loss = args.use_huber_loss  # 是否使用Huber损失，一种在回归任务中更稳健的损失函数
        self._use_valuenorm = args.use_valuenorm  # 是否使用值归一化，帮助保持值函数的稳定性
        self._use_value_active_masks = args.use_value_active_masks  # 是否使用值函数的活动掩码，限制只有活跃的智能体才计算损失，与active_mask合用
        self._use_policy_active_masks = args.use_policy_active_masks  # 是否使用策略的活动掩码，限制只有活跃的智能体才进行策略更新，与active_mask合用
        self.dec_actor = args.dec_actor  # True表示直接生成动作，不需要复杂解码步骤逐步生成动作。默认False
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss. 用于训练Critic网络
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timestep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample
        # adv_targ：优势函数，论文中的A

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        old_action_log_probs_batch = old_action_log_probs_batch.view(-1, 1)
        adv_targ = check(adv_targ).to(**self.tpdv)
        adv_targ = adv_targ.view(-1, 1)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        value_preds_batch = value_preds_batch.view(-1, 1)
        return_batch = check(return_batch).to(**self.tpdv)
        return_batch = return_batch.view(-1, 1)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        active_masks_batch = active_masks_batch.view(-1, 1)


        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)  # 新老策略的比值，论文中公式（5-2）

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        # print('adv:', advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            # print('ppo_epoch:', _)
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)
            # print(data_generator)
            for sample in data_generator:
                # print(sample)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
