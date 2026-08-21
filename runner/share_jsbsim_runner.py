import logging
import os
import time
from typing import List

import numpy as np
import torch

from algorithms.utils.buffer import SharedReplayBuffer
from .base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class ShareJSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay  # type: bool

        # policy & algorithm
        if self.algorithm_name == "mappo":
            from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        if self.use_selfplay:
            self.buffer = SharedReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.share_obs_space, self.act_space)
        else:
            self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.share_obs_space, self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        if self.use_selfplay:

            from algorithms.utils.selfplay import get_algorithm
            self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)
            self.init_elo = self.all_args.init_elo
            self.latest_elo = self.init_elo

            assert self.all_args.n_choose_opponents <= self.n_rollout_threads, \
                "Number of different opponents({}) must less than or equal to number of training threads({})!" \
                .format(self.all_args.n_choose_opponents, self.n_rollout_threads)
            self.policy_pool = {}  # type: dict[str, float]
            self.policy_snapshots = {}
            self.current_opponent_ids = []
            self.opponent_policy = [
                Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
                for _ in range(self.all_args.n_choose_opponents)]
            self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
            self.opponent_obs = np.zeros_like(self.buffer.obs[0])
            self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
            self.opponent_masks = np.ones_like(self.buffer.masks[0])

            self.eval_opponent_policy = Policy(
                self.all_args,
                self.obs_space,
                self.share_obs_space,
                self.act_space,
                device=self.device,
            )

            logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                         .format(self.all_args.selfplay_algorithm, self.all_args.n_choose_opponents))

        if self.model_dir is not None:
            self.restore()
        if self.use_selfplay:
            self._initialize_opponents()

    @staticmethod
    def _clone_actor_state(actor):
        return {
            key: value.detach().cpu().clone()
            for key, value in actor.state_dict().items()
        }

    def _store_policy_snapshot(self, snapshot_id, elo):
        snapshot_id = str(snapshot_id)
        state = self._clone_actor_state(self.policy.actor)
        self.policy_snapshots[snapshot_id] = state
        self.policy_pool[snapshot_id] = float(elo)
        torch.save(state, str(self.save_dir) + f"/actor_{snapshot_id}.pt")

    def _load_policy_snapshot(self, policy, snapshot_id):
        snapshot_id = str(snapshot_id)
        if snapshot_id == "latest":
            policy.actor.load_state_dict(self.policy.actor.state_dict())
            policy.prep_rollout()
            return
        state = self.policy_snapshots.get(snapshot_id)
        if state is None:
            for root in (self.model_dir, self.save_dir):
                if root is None:
                    continue
                path = str(root) + f"/actor_{snapshot_id}.pt"
                if os.path.isfile(path):
                    loaded = torch.load(
                        path, map_location=self.device, weights_only=True
                    )
                    state = {
                        key: value.detach().cpu().clone()
                        for key, value in loaded.items()
                    }
                    self.policy_snapshots[snapshot_id] = state
                    break
        if state is None:
            raise FileNotFoundError(f"Missing actor snapshot {snapshot_id!r}")
        policy.actor.load_state_dict(state)
        policy.prep_rollout()

    def _initialize_opponents(self):
        if not self.policy_pool:
            self._store_policy_snapshot("initial", self.init_elo)
        elif not self.policy_snapshots:
            for snapshot_id in self.policy_pool:
                self._load_policy_snapshot(self.opponent_policy[0], snapshot_id)

        if len(self.current_opponent_ids) != len(self.opponent_policy):
            self.current_opponent_ids = [
                self.selfplay_algo.choose(self.policy_pool)
                for _ in self.opponent_policy
            ]
        for policy, snapshot_id in zip(
            self.opponent_policy, self.current_opponent_ids
        ):
            self._load_policy_snapshot(policy, snapshot_id)

    def update_opponent_pool(self, episode):
        self._store_policy_snapshot(str(episode), self.latest_elo)
        self.reset_opponent()

    def run(self):
        self.warmup()

        start = time.time()
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(self.start_episode, episodes):

            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    terminated,
                    truncated,
                    infos,
                ) = self.envs.step(actions)

                data = (
                    obs,
                    share_obs,
                    actions,
                    rewards,
                    terminated,
                    truncated,
                    infos,
                    action_log_probs,
                    values,
                    rnn_states_actor,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos.update(self.rollout_reward_metrics())
                if "average_episode_rewards" in train_infos:
                    logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_info(train_infos, self.total_num_steps)

            # Refresh the training population independently from evaluation.
            if self.use_selfplay and episode % self.selfplay_interval == 0:
                self.update_opponent_pool(episode)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model and all resume state after population/Elo updates.
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # [Selfplay] divide ego/opponent of initial obs
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                      np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        if self.use_selfplay:
            opponent_actions = np.zeros_like(actions)
            for policy_idx, policy in enumerate(self.opponent_policy):
                env_idx = self.opponent_env_split[policy_idx]
                opponent_action, opponent_rnn_states \
                    = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                                 np.concatenate(self.opponent_rnn_states[env_idx]),
                                 np.concatenate(self.opponent_masks[env_idx]))
                opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
                self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
            actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data: List[np.ndarray]):
        (
            obs,
            share_obs,
            actions,
            rewards,
            terminated,
            truncated,
            infos,
            action_log_probs,
            values,
            rnn_states_actor,
            rnn_states_critic,
        ) = data
        rewards = self._bootstrap_time_limits(
            rewards, truncated, infos, rnn_states_critic
        )
        dones = np.logical_or(terminated, truncated).squeeze(axis=-1)
        dones_env = np.all(dones, axis=-1)
        controlled_agents = self.buffer.num_agents
        controlled_dones = dones[:, :controlled_agents]

        rnn_states_actor[controlled_dones] = 0.0
        rnn_states_critic[controlled_dones] = 0.0

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones] = 0.0
        bad_masks = np.ones_like(masks)
        bad_masks[truncated.squeeze(axis=-1)] = 0.0

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        # [Selfplay] divide ego/opponent of collecting data TODO: shared_obs
        if self.use_selfplay:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            self.opponent_masks = masks[:, self.num_agents // 2:, ...]
            opponent_dones = dones[:, self.num_agents // 2:]
            self.opponent_rnn_states[opponent_dones] = 0.0

            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
            actions = actions[:, :self.num_agents // 2, ...]
            rewards = rewards[:, :self.num_agents // 2, ...]
            masks = masks[:, :self.num_agents // 2, ...]
            bad_masks = bad_masks[:, :self.num_agents // 2, ...]
            active_masks = active_masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values, \
            rnn_states_actor, rnn_states_critic, bad_masks=bad_masks,
            active_masks=active_masks)

    @torch.no_grad()
    def _bootstrap_time_limits(self, rewards, truncated, infos, rnn_states_critic):
        if not self.all_args.use_proper_time_limits:
            return rewards
        rewards = rewards.copy()
        controlled_agents = self.buffer.num_agents
        for env_id, info in enumerate(infos):
            truncated_agents = truncated[env_id, :controlled_agents, 0]
            if not np.any(truncated_agents):
                continue
            final_share_obs = np.asarray(info["final_share_observation"])[
                :controlled_agents
            ]
            bootstrap_values = self.policy.get_values(
                final_share_obs,
                rnn_states_critic[env_id],
                np.ones((controlled_agents, 1), dtype=np.float32),
            )
            bootstrap_values = _t2n(bootstrap_values)
            agent_ids = np.flatnonzero(truncated_agents)
            rewards[env_id, agent_ids, 0] += (
                self.all_args.gamma * bootstrap_values[agent_ids, 0]
            )
        return rewards

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        # [Selfplay] Choose opponent policy for evaluation
        if self.use_selfplay:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.all_args.n_choose_opponents)]
            assert self.eval_episodes >= self.all_args.n_choose_opponents, \
            f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.all_args.n_choose_opponents}"
            eval_each_episodes = self.eval_episodes // self.all_args.n_choose_opponents
            eval_cur_opponent_idx = 0
            eval_active_opponent = None
            eval_opponent_cumulative_rewards = np.zeros_like(
                eval_cumulative_rewards
            )
            eval_opponent_episode_rewards = []
            eval_episode_opponent_ids = []
            logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if (
                self.use_selfplay
                and eval_cur_opponent_idx < len(eval_choose_opponents)
                and total_episodes >= eval_cur_opponent_idx * eval_each_episodes
            ):
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self._load_policy_snapshot(self.eval_opponent_policy, policy_idx)
                eval_active_opponent = policy_idx
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes+1}/{self.eval_episodes})")

                # reset obs/rnn/mask
                eval_obs, eval_share_obs, _ = self.eval_envs.reset()
                eval_masks = np.ones_like(eval_masks, dtype=np.float32)
                eval_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]
                eval_opponent_masks = np.ones_like(eval_masks, dtype=np.float32)
                eval_opponent_rnn_states = np.zeros_like(eval_rnn_states, dtype=np.float32)
                eval_cumulative_rewards.fill(0)
                eval_opponent_cumulative_rewards.fill(0)

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # [Selfplay] get actions of opponent policy
            if self.use_selfplay:
                eval_opponent_actions, eval_opponent_rnn_states \
                    = self.eval_opponent_policy.act(np.concatenate(eval_opponent_obs),
                                                    np.concatenate(eval_opponent_rnn_states),
                                                    np.concatenate(eval_opponent_masks),
                                                    deterministic=True)
                eval_opponent_rnn_states = np.array(np.split(_t2n(eval_opponent_rnn_states), self.n_eval_rollout_threads))
                eval_opponent_actions = np.array(np.split(_t2n(eval_opponent_actions), self.n_eval_rollout_threads))
                eval_actions = np.concatenate((eval_actions, eval_opponent_actions), axis=1)

            # Obser reward and next obs
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_terminated,
                eval_truncated,
                eval_infos,
            ) = self.eval_envs.step(eval_actions)
            eval_dones = np.logical_or(eval_terminated, eval_truncated)

            # [Selfplay] get ego reward
            if self.use_selfplay:
                eval_opponent_rewards = eval_rewards[:, self.num_agents // 2:, ...]
                eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]

            eval_cumulative_rewards += eval_rewards
            if self.use_selfplay:
                eval_opponent_cumulative_rewards += eval_opponent_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            if self.use_selfplay and np.any(eval_dones_env):
                eval_opponent_episode_rewards.append(
                    eval_opponent_cumulative_rewards[eval_dones_env == True]
                )
                eval_episode_opponent_ids.extend(
                    [eval_active_opponent] * int(np.sum(eval_dones_env))
                )
            eval_cumulative_rewards[eval_dones_env == True] = 0
            if self.use_selfplay:
                eval_opponent_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)
            # [Selfplay] reset opponent mask/rnn_states
            if self.use_selfplay:
                eval_opponent_obs = eval_obs[:, self.num_agents // 2:, ...]
                eval_obs = eval_obs[:, :self.num_agents // 2, ...]
                eval_opponent_masks[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), *eval_opponent_masks.shape[1:]), dtype=np.float32)
                eval_opponent_rnn_states[eval_dones_env == True] = \
                    np.zeros(((eval_dones_env == True).sum(), *eval_opponent_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean() 
        if self.use_selfplay:
            ego_episode_rewards = np.concatenate(eval_episode_rewards).mean(
                axis=(1, 2)
            )
            opponent_episode_rewards = np.concatenate(
                eval_opponent_episode_rewards
            ).mean(axis=(1, 2))
            reward_difference = ego_episode_rewards - opponent_episode_rewards
            actual_scores = np.full_like(reward_difference, 0.5, dtype=np.float64)
            actual_scores[reward_difference > 100] = 1.0
            actual_scores[reward_difference < -100] = 0.0
            previous_elo = self.latest_elo
            self.latest_elo = self.selfplay_algo.update(
                self.policy_pool,
                {
                    "ego_elo": self.latest_elo,
                    "opponent_ids": eval_episode_opponent_ids,
                    "actual_scores": actual_scores.tolist(),
                },
            )
            eval_infos["eval_opponent_average_episode_rewards"] = float(
                opponent_episode_rewards.mean()
            )
            eval_infos["eval_elo_gain"] = self.latest_elo - previous_elo
            eval_infos["latest_elo"] = self.latest_elo
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        self.log_info(eval_infos, total_num_steps)

        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index
        render_episode_rewards = 0
        render_obs, render_share_obs, _ = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        if self.use_selfplay:
            policy_idx = self.render_opponent_index
            self._load_policy_snapshot(self.eval_opponent_policy, policy_idx)
            # reset obs/rnn/mask
            render_obs, render_share_obs, _ = self.envs.reset()
            render_masks = np.ones_like(render_masks, dtype=np.float32)
            render_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
            render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
            render_obs = render_obs[:, :self.num_agents // 2, ...]
            render_opponent_masks = np.ones_like(render_masks, dtype=np.float32)
            render_opponent_rnn_states = np.zeros_like(render_rnn_states, dtype=np.float32)
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            
            # [Selfplay] get actions of opponent policy
            if self.use_selfplay:
                render_opponent_actions, render_opponent_rnn_states \
                    = self.eval_opponent_policy.act(np.concatenate(render_opponent_obs),
                                                    np.concatenate(render_opponent_rnn_states),
                                                    np.concatenate(render_opponent_masks),
                                                    deterministic=True)
                render_opponent_actions = np.expand_dims(_t2n(render_opponent_actions), axis=0)
                render_opponent_rnn_states = np.expand_dims(_t2n(render_opponent_rnn_states), axis=0)
                render_actions = np.concatenate((render_actions, render_opponent_actions), axis=1)
            # Obser reward and next obs
            (
                render_obs,
                render_share_obs,
                render_rewards,
                render_terminated,
                render_truncated,
                render_infos,
            ) = self.envs.step(render_actions)
            render_dones = np.logical_or(render_terminated, render_truncated)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
            if self.use_selfplay:
                render_opponent_obs = render_obs[:, self.num_agents // 2:, ...]
                render_obs = render_obs[:, :self.num_agents // 2, ...]

        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        super().save(episode)

    def reset_opponent(self, reset_environment=True):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            self._load_policy_snapshot(policy, choose_idx)
        self.current_opponent_ids = list(choose_opponents)
        logging.info(f" Choose opponents {choose_opponents} for training")

        if not reset_environment:
            return

        # clear buffer
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        # reset env
        obs, share_obs, _ = self.envs.reset()
        if self.all_args.n_choose_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()
