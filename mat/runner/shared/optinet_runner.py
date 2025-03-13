import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()

class OptinetRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(OptinetRunner, self).__init__(config)

    def run(self, episode, episodes):
        block_flag = False
        self.warmup()

        start = time.time()

        train_episode_rewards = [0]
        done_episodes_rewards = []

        train_episode_scores = [0]
        done_episodes_scores = []

        if self.use_linear_lr_decay:
            self.trainer.policy.lr_decay(episode, episodes)

        for step in range(self.episode_length):
            # Sample actions
            # print('step:', step)
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

            # Obser reward and next obs

            obs, share_obs, rewards, dones, infos, available_actions, blocked_allocation \
                = self.envs.make_step(actions, step, self.episode_length)

            dones_env = np.all(dones)
            reward_env = np.mean(rewards).flatten()
            train_episode_rewards += reward_env

            score_env = [t_info[0]["score_reward"] for t_info in infos]
            train_episode_scores += np.array(score_env)
            if dones_env:
                done_episodes_rewards.append(train_episode_rewards)
                train_episode_rewards = 0
                done_episodes_scores.append(train_episode_scores)
                train_episode_scores = 0

            data = obs, share_obs, rewards, dones, infos, available_actions, \
                   values, actions, action_log_probs, \
                   rnn_states, rnn_states_critic, self.envs.active_masks

            # insert data into buffer
            self.insert(data)

            if blocked_allocation:
                block_flag = True
            if dones_env:
                break

        # compute return and update network
        self.compute()
        train_infos = self.train()

        # post process
        total_num_steps = (episode + 1) * self.episode_length
        # save model
        if (episode % self.save_interval == 0 or episode == episodes - 1):
            self.save(episode)

        # log information
        if episode % self.log_interval == 0:
            end = time.time()
            print("\n Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.algorithm_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))

            self.log_train(train_infos, total_num_steps)

            if len(done_episodes_rewards) > 0:
                aver_episode_rewards = np.mean(done_episodes_rewards)
                self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                done_episodes_rewards = []

                aver_episode_scores = np.mean(done_episodes_scores)
                self.writter.add_scalars("train_episode_scores", {"aver_scores": aver_episode_scores}, total_num_steps)
                done_episodes_scores = []
                print("some episodes done, average rewards: {}, scores: {}"
                      .format(aver_episode_rewards, aver_episode_scores))
        return self.envs.topology, self.envs.service_dict, block_flag
            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, ava = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = ava.copy()
        self.buffer.active_masks[0] = self.envs.active_masks.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(self.buffer.share_obs[step],
                                              self.buffer.obs[step],
                                              self.buffer.rnn_states[step],
                                              self.buffer.rnn_states_critic[step],
                                              self.buffer.available_actions[step])
        # [self.env, agents, dim]
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))
        rnn_states = np.array(_t2n(rnn_state))
        rnn_states_critic = np.array(_t2n(rnn_state_critic))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, active_masks = data

        dones_env = np.all(dones)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks = np.ones((self.num_agents, 1), dtype=np.float32)
        # active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]
        eval_episode_scores = []
        one_episode_scores = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, ava = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.all_args.eval_episodes, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(ava),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.all_args.eval_episodes))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.all_args.eval_episodes))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava = self.eval_envs.step(eval_actions)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_scores = [t_info[0]["score_reward"] for t_info in eval_infos]
            one_episode_scores += np.array(eval_scores)

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.all_args.eval_episodes):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

                    eval_episode_scores.append(one_episode_scores[eval_i])
                    one_episode_scores[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = '/eval_average_episode_rewards'
                key_max = '/eval_max_episode_rewards'
                key_scores = '/eval_average_episode_scores'
                eval_env_infos = {key_average: eval_episode_rewards,
                                  key_max: [np.max(eval_episode_rewards)],
                                  key_scores: eval_episode_scores}
                self.log_env(eval_env_infos, total_num_steps)

                print("eval average episode rewards: {}, scores: {}."
                      .format(np.mean(eval_episode_rewards), np.mean(eval_episode_scores)))
                break
