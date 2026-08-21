from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from algorithms.utils.buffer import ReplayBuffer, SharedReplayBuffer
from config import get_config
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv
from runner.jsbsim_runner import JSBSimRunner
from runner.share_jsbsim_runner import ShareJSBSimRunner


class _TruncatingEnv:
    observation_space = gym.spaces.Box(low=-100, high=100, shape=(1,))
    action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        return np.array([0.0], dtype=np.float32), {"reset": True}

    def step(self, action):
        return (
            np.array([42.0], dtype=np.float32),
            1.0,
            False,
            True,
            {"TimeLimit.truncated": True},
        )

    def close(self):
        pass


class _SharedTruncatingEnv(_TruncatingEnv):
    share_observation_space = gym.spaces.Box(low=-100, high=100, shape=(1,))

    def __init__(self):
        self.state = np.array([0.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.state = np.array([0.0], dtype=np.float32)
        return self.state.copy(), {"reset": True}

    def step(self, action):
        self.state = np.array([84.0], dtype=np.float32)
        return self.state / 2, 1.0, False, True, {"TimeLimit.truncated": True}

    def get_state(self):
        return self.state.copy()


def test_vec_env_preserves_final_observation_when_auto_resetting():
    envs = DummyVecEnv([_TruncatingEnv])
    try:
        obs, reset_infos = envs.reset()
        assert obs[0, 0] == 0
        obs, rewards, terminated, truncated, infos = envs.step([0])
        assert obs[0, 0] == 0
        assert not terminated[0]
        assert truncated[0]
        np.testing.assert_array_equal(infos[0]["final_observation"], [42.0])
    finally:
        envs.close()


def test_shared_vec_env_preserves_final_local_and_central_observations():
    envs = ShareDummyVecEnv([_SharedTruncatingEnv])
    try:
        obs, share_obs, _ = envs.reset()
        result = envs.step([0])
        obs, share_obs, _, terminated, truncated, infos = result
        assert not terminated[0]
        assert truncated[0]
        np.testing.assert_array_equal(infos[0]["final_observation"], [42.0])
        np.testing.assert_array_equal(infos[0]["final_share_observation"], [84.0])
        np.testing.assert_array_equal(obs[0], [0.0])
        np.testing.assert_array_equal(share_obs[0], [0.0])
    finally:
        envs.close()


class _ConstantValuePolicy:
    def __init__(self, value):
        self.value = value

    def get_values(self, obs, rnn_states, masks):
        return torch.full((len(obs), 1), self.value, dtype=torch.float32)


def test_ppo_timeout_bootstraps_from_final_observation():
    runner = JSBSimRunner.__new__(JSBSimRunner)
    runner.all_args = SimpleNamespace(use_proper_time_limits=True, gamma=0.99)
    runner.buffer = SimpleNamespace(num_agents=1)
    runner.policy = _ConstantValuePolicy(5.0)
    rewards = np.zeros((1, 1, 1), dtype=np.float32)
    truncated = np.ones((1, 1, 1), dtype=bool)
    infos = np.array([{"final_observation": np.array([[42.0]])}], dtype=object)
    rnn = np.zeros((1, 1, 1, 2), dtype=np.float32)

    corrected = runner._bootstrap_time_limits(rewards, truncated, infos, rnn)
    assert corrected[0, 0, 0] == np.float32(0.99 * 5.0)


def test_mappo_timeout_bootstraps_from_final_central_observation():
    runner = ShareJSBSimRunner.__new__(ShareJSBSimRunner)
    runner.all_args = SimpleNamespace(use_proper_time_limits=True, gamma=0.9)
    runner.buffer = SimpleNamespace(num_agents=2)
    runner.policy = _ConstantValuePolicy(7.0)
    rewards = np.zeros((1, 2, 1), dtype=np.float32)
    truncated = np.ones((1, 2, 1), dtype=bool)
    infos = np.array(
        [{"final_share_observation": np.array([[84.0], [84.0]])}],
        dtype=object,
    )
    rnn = np.zeros((1, 2, 1, 2), dtype=np.float32)

    corrected = runner._bootstrap_time_limits(rewards, truncated, infos, rnn)
    np.testing.assert_allclose(corrected[0, :, 0], 0.9 * 7.0)


def test_corrected_timeout_reward_is_not_replaced_by_bad_mask():
    args = get_config().parse_args([])
    args.buffer_size = 1
    args.n_rollout_threads = 1
    args.use_proper_time_limits = True
    args.use_gae = False
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    buffer = ReplayBuffer(args, 1, obs_space, gym.spaces.Discrete(2))
    buffer.rewards[0, 0, 0, 0] = 4.95
    buffer.masks[1, 0, 0, 0] = 0
    buffer.bad_masks[1, 0, 0, 0] = 0

    buffer.compute_returns(np.zeros((1, 1, 1), dtype=np.float32))

    assert buffer.returns[0, 0, 0, 0] == np.float32(4.95)


def test_shared_buffer_propagates_bad_masks_to_parent_storage():
    args = get_config().parse_args([])
    args.buffer_size = 1
    args.n_rollout_threads = 1
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    buffer = SharedReplayBuffer(
        args, 1, obs_space, obs_space, gym.spaces.Discrete(2)
    )
    scalar = np.zeros((1, 1, 1), dtype=np.float32)
    rnn = np.zeros((1, 1, args.recurrent_hidden_layers, args.recurrent_hidden_size))
    buffer.insert(
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        rnn,
        rnn,
        bad_masks=scalar,
    )
    assert buffer.bad_masks[1, 0, 0, 0] == 0
