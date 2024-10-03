"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
import wandb
from copy import deepcopy
from runner.shared.base_runner import Runner
from algorithms.utils.util import AsynchControl

# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.asynch_control = AsynchControl(num_envs=self.n_eval_rollout_threads, num_agents=self.num_agents)

    def run(self):
        wandb.init(project="Async-MAPPO", config={
            "algorithm":"Async-MAPPO",
            "environment":"Async-Truck",
            "num_episodes":1000,
        })

        self.warmup()

        start = time.time()
        episodes = int(self.total_episode) // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length * self.num_agents):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.async_collect(active_agents=self.asynch_control.active_agents())

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # Update activate agent
                self.asynch_control.step(obs, actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions_env,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                # self.insert(data)
                self.async_insert(data, active_agents=self.asynch_control.active_agents(), p_agents=self.asynch_control.previous_agents())
                tmp_done = np.array([all(done_dict.values()) for done_dict in dones])
                if tmp_done[0]:
                    break

            self.warmup()
            # compute return and update network
            # self.buffer.update_mask(self.asynch_control.cnt)
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            wandb.log({
                "episode": episode,
                "episode_reward": np.mean(self.buffer.rewards) * self.episode_length,
                "episode_length": self.episode_length,
                "total_timesteps": total_num_steps,
            })

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        wandb.finish()

    # def warmup(self):
    #     # reset env
    #     obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

    #     # replay buffer
    #     if self.use_centralized_V:
    #         share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]
    #         share_obs = np.expand_dims(share_obs, 1).repeat(
    #             self.num_agents, axis=1
    #         )  # shape = shape = [env_num, agent_num， agent_num * obs_dim]
    #     else:
    #         share_obs = obs

    #     self.buffer.share_obs[0] = share_obs.copy()
    #     self.buffer.obs[0] = obs.copy()
    

    def warmup(self):
        # reset env
        self.asynch_control.reset()
        obs = self.envs.reset()
        self.obs = obs
        # replay buffer
        for e, a, step in self.asynch_control.active_agents():
            share_obs = np.concatenate(list(obs[e].values()))
            self.buffer.obs[step, e, a] = obs[e][a].copy()
            self.buffer.share_obs[step, e, a] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def async_collect(self, active_agents):
        self.trainer.prep_rollout()

        concat_share_obs = np.stack([self.buffer.share_obs[step, e, a] for e, a, step in active_agents],  axis=0)
        concat_obs = np.stack([self.buffer.obs[step, e, a] for e, a, step in active_agents],  axis=0)

        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            concat_share_obs,
            concat_obs,
            np.concatenate([self.buffer.rnn_states[step, e, a] for e, a, step in active_agents],  axis=0),
            np.concatenate([self.buffer.rnn_states_critic[step, e, a] for e, a, step in active_agents],  axis=0),
            np.concatenate([self.buffer.masks[step, e, a] for e, a, step in active_agents],  axis=0),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            # [self.envs{agents_id: action}]
            actions_env = [{} for _ in range(self.n_rollout_threads)]
            value_env = deepcopy(actions_env)
            action_log_probs_env = deepcopy(actions_env)
            for i, (e, a, step) in enumerate(active_agents):
                actions_env[e][a] = actions[e,i,:]
                value_env[e][a] = values[e,i,:]
                action_log_probs_env[e][a] = action_log_probs[e,i,:]

        return (
            value_env,
            actions,
            action_log_probs_env,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )
    
    def async_insert(self, data, active_agents=None, p_agents=None):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        tmp_done = np.array([all(done_dict.values()) for done_dict in dones])
        # dones_env = np.array(
        #     [[True for _ in range(rnn_states.shape[1])] if tmp_done[e] else [False for _ in range(rnn_states.shape[1])]
        #      for e in range(len(tmp_done))]
        #     )
        mask_done = np.array(
            [[True for _ in range(self.num_agents)] if tmp_done[e] else [False for _ in range(self.num_agents)]
             for e in range(len(tmp_done))]
            )
        tmp_rnn_states = np.zeros([self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size])
        tmp_rnn_states_critic = np.zeros([self.n_rollout_threads, self.num_agents, *self.buffer.rnn_states_critic.shape[3:]])
        # for e, a, step in active_agents:
        #     if not tmp_done[e]:
        #         tmp_rnn_states[e, a] = rnn_states[e, a].copy()
        #         tmp_rnn_states_critic[e, a] = rnn_states_critic[e, a].copy()

        # rnn_states[dones_env == True] = np.zeros(
        #     ((dones_env == True).sum(), self.hidden_size),
        #     dtype=np.float32,
        # )
        # rnn_states_critic[dones_env == True] = np.zeros(
        #     ((dones_env == True).sum(), *self.buffer.rnn_states_critic.shape[4:]),
        #     dtype=np.float32,
        # )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[mask_done == True] = np.zeros(((mask_done == True).sum(), 1), dtype=np.float32)
        share_obs = deepcopy(obs)
        concat_share_obs = deepcopy(obs)
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                if a not in share_obs[e].keys():
                    step = self.asynch_control.cnt[e,a]
                    share_obs[e][a] = self.buffer.obs[step, e, a].copy()

        for i, (e, a, step) in enumerate(active_agents):
            concat_share_obs[e][a] = np.concatenate([share_obs[e][key] for key in range(self.num_agents)])
        for i, (e, a, step) in enumerate(p_agents):
            if not tmp_done[e]:
                tmp_rnn_states[e, a] = rnn_states[e, i].copy()
                tmp_rnn_states_critic[e, a] = rnn_states_critic[e, i].copy()
        
        self.buffer.async_insert(
            concat_share_obs,
            obs,
            tmp_rnn_states,
            tmp_rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_agents= active_agents,
            p_agents=p_agents,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                        eval_actions[:, :, i]
                    ]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
