from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from algorithms.mappo.ppo_trainer import PPOTrainer as MAPPOTrainer
from algorithms.ppo.ppo_actor import PPOActor
from algorithms.utils.act import ACTLayer
from algorithms.utils.buffer import ReplayBuffer, SharedReplayBuffer
from config import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


def _buffer_args(buffer_size=6, n_rollout_threads=2):
    args = get_config().parse_args(args=[])
    args.buffer_size = buffer_size
    args.n_rollout_threads = n_rollout_threads
    args.recurrent_hidden_layers = 1
    args.recurrent_hidden_size = 4
    return args


def _fill_unique_observations(buffer):
    for step in range(buffer.buffer_size):
        for env_id in range(buffer.n_rollout_threads):
            for agent_id in range(buffer.num_agents):
                unique_id = (
                    step
                    + buffer.buffer_size * env_id
                    + buffer.buffer_size * buffer.n_rollout_threads * agent_id
                )
                buffer.obs[step, env_id, agent_id, 0] = unique_id


def _generated_observation_ids(generator, shared=False):
    ids = []
    for sample in generator:
        obs_batch = sample[0]
        ids.extend(obs_batch[:, 0].astype(np.int64).tolist())
        if shared:
            assert sample[5].shape[-1] == 1
    return ids


def test_shoot_action_sampling_and_evaluation_use_same_joint_distribution():
    args = get_config().parse_args(args=[])
    args.use_prior = False
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(18,))
    act_space = gym.spaces.Tuple(
        (gym.spaces.MultiDiscrete([3, 5, 3]), gym.spaces.Discrete(2))
    )
    actor = PPOActor(args, obs_space, act_space, device=torch.device("cpu"))
    batch_size = 8
    obs = np.stack([obs_space.sample() for _ in range(batch_size)])
    masks = np.ones((batch_size, 1), dtype=np.float32)
    rnn_states = np.zeros(
        (batch_size, actor.recurrent_hidden_layers, actor.recurrent_hidden_size),
        dtype=np.float32,
    )

    with torch.no_grad():
        actions, sampled_log_probs, _ = actor(obs, rnn_states, masks)
        evaluated_log_probs, _ = actor.evaluate_actions(
            obs, rnn_states, actions, masks
        )

    torch.testing.assert_close(sampled_log_probs, evaluated_log_probs)
    torch.testing.assert_close(
        torch.exp(evaluated_log_probs - sampled_log_probs),
        torch.ones_like(sampled_log_probs),
    )


def test_prior_is_rejected_for_non_shooting_action_spaces():
    args = get_config().parse_args(args=[])
    args.use_prior = True
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(5,))

    with pytest.raises(ValueError, match="shoot"):
        PPOActor(args, obs_space, gym.spaces.Discrete(2), device=torch.device("cpu"))


def test_entropy_is_per_sample_and_independent_of_batch_size():
    layer = ACTLayer(
        gym.spaces.MultiDiscrete([3, 4]),
        input_dim=5,
        hidden_size=[],
        activation_id=1,
        gain=0.01,
    )
    one_input = torch.zeros((1, 5))
    one_action = torch.zeros((1, 2), dtype=torch.long)
    many_inputs = one_input.repeat(7, 1)
    many_actions = one_action.repeat(7, 1)

    _, one_entropy = layer.evaluate_actions(one_input, one_action)
    _, many_entropy = layer.evaluate_actions(many_inputs, many_actions)

    torch.testing.assert_close(many_entropy, one_entropy.repeat(7, 1))


def test_replay_generator_covers_every_env_agent_timestep_once():
    args = _buffer_args()
    obs_space = gym.spaces.Box(low=-1, high=1000, shape=(1,))
    act_space = gym.spaces.Discrete(2)
    buffer = ReplayBuffer(args, 3, obs_space, act_space)
    _fill_unique_observations(buffer)

    ids = _generated_observation_ids(
        ReplayBuffer.recurrent_generator(buffer, num_mini_batch=4, data_chunk_length=2)
    )
    expected = list(range(buffer.buffer_size * buffer.n_rollout_threads * buffer.num_agents))
    assert sorted(ids) == expected


def test_shared_generator_covers_every_env_agent_timestep_once():
    args = _buffer_args()
    obs_space = gym.spaces.Box(low=-1, high=1000, shape=(1,))
    share_obs_space = gym.spaces.Box(low=-1, high=1000, shape=(2,))
    act_space = gym.spaces.MultiDiscrete([3, 4, 2, 2])
    buffer = SharedReplayBuffer(args, 3, obs_space, share_obs_space, act_space)
    _fill_unique_observations(buffer)

    ids = _generated_observation_ids(
        buffer.recurrent_generator(
            buffer.advantages, num_mini_batch=4, data_chunk_length=2
        ),
        shared=True,
    )
    expected = list(range(buffer.buffer_size * buffer.n_rollout_threads * buffer.num_agents))
    assert sorted(ids) == expected


@pytest.mark.parametrize("buffer_class", [ReplayBuffer, SharedReplayBuffer])
def test_recurrent_generator_rejects_chunks_that_cross_trajectory_boundaries(
    buffer_class,
):
    args = _buffer_args(buffer_size=5)
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    act_space = gym.spaces.Discrete(2)
    if buffer_class is ReplayBuffer:
        buffer = buffer_class(args, 2, obs_space, act_space)
        generator = buffer.recurrent_generator(buffer, 2, 2)
    else:
        buffer = buffer_class(args, 2, obs_space, obs_space, act_space)
        generator = buffer.recurrent_generator(buffer.advantages, 2, 2)

    with pytest.raises(ValueError, match="divisible"):
        next(generator)


def test_shared_buffer_stores_one_joint_log_probability():
    args = _buffer_args()
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    act_space = gym.spaces.MultiDiscrete([3, 4, 2, 2])
    buffer = SharedReplayBuffer(args, 2, obs_space, obs_space, act_space)
    assert buffer.action_log_probs.shape == (
        args.buffer_size,
        args.n_rollout_threads,
        2,
        1,
    )


def test_shared_buffer_rejects_broadcast_log_probabilities():
    args = _buffer_args()
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    act_space = gym.spaces.MultiDiscrete([3, 4, 2, 2])
    buffer = SharedReplayBuffer(args, 2, obs_space, obs_space, act_space)
    nenv, nagent = args.n_rollout_threads, 2
    obs = np.zeros((nenv, nagent, 3), dtype=np.float32)
    actions = np.zeros((nenv, nagent, 4), dtype=np.float32)
    scalars = np.zeros((nenv, nagent, 1), dtype=np.float32)
    rnn = np.zeros((nenv, nagent, 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="action_log_probs"):
        buffer.insert(
            obs,
            obs,
            actions,
            scalars,
            scalars,
            np.zeros((nenv, nagent, 4), dtype=np.float32),
            scalars,
            rnn,
            rnn,
        )


def test_shared_advantage_normalization_uses_only_active_samples():
    args = _buffer_args(buffer_size=2, n_rollout_threads=1)
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    buffer = SharedReplayBuffer(args, 2, obs_space, obs_space, gym.spaces.Discrete(2))
    buffer.returns[:-1, 0, :, 0] = np.array([[1.0, 1e6], [3.0, -1e6]])
    buffer.value_preds[:-1] = 0
    buffer.active_masks[:-1, 0, :, 0] = np.array([[1.0, 0.0], [1.0, 0.0]])

    advantages = buffer.advantages
    np.testing.assert_allclose(advantages[:, 0, 0, 0], [-1.0, 1.0], atol=1e-4)
    np.testing.assert_array_equal(advantages[:, 0, 1, 0], 0.0)


class _MaskedPolicy:
    def __init__(self, args):
        self.actor = nn.Linear(1, 1, bias=False)
        self.critic = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.critic.weight)
        self.optimizer = torch.optim.SGD(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.01
        )

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        masks,
        active_masks=None,
    ):
        inputs = torch.ones((len(obs), 1), dtype=torch.float32)
        values = self.critic(inputs)
        log_probs = self.actor(inputs)
        entropy = torch.ones_like(log_probs) + 0.0 * log_probs
        return values, log_probs, entropy


def _mappo_sample(inactive_advantage, inactive_return):
    zeros = np.zeros((2, 1), dtype=np.float32)
    rnn = np.zeros((2, 1, 1), dtype=np.float32)
    return (
        zeros,
        zeros,
        zeros,
        np.ones((2, 1), dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
        zeros,
        np.array([[1.0], [inactive_advantage]], dtype=np.float32),
        np.array([[1.0], [inactive_return]], dtype=np.float32),
        zeros,
        rnn,
        rnn,
    )


def test_mappo_losses_ignore_inactive_samples():
    args = get_config().parse_args(args=[])
    args.use_max_grad_norm = False
    trainer = MAPPOTrainer(args, device=torch.device("cpu"))

    baseline = trainer.ppo_update(_MaskedPolicy(args), _mappo_sample(0.0, 0.0))
    extreme = trainer.ppo_update(
        _MaskedPolicy(args), _mappo_sample(1e6, -1e6)
    )

    torch.testing.assert_close(baseline[0], extreme[0])
    torch.testing.assert_close(baseline[1], extreme[1])


class _InsertCapture:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.args = None

    def insert(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_share_runner_resets_rnn_state_when_one_agent_dies():
    runner = ShareJSBSimRunner.__new__(ShareJSBSimRunner)
    runner.n_rollout_threads = 1
    runner.num_agents = 4
    runner.use_selfplay = False
    runner.all_args = SimpleNamespace(use_proper_time_limits=False)
    runner.buffer = _InsertCapture(num_agents=4)
    obs = np.zeros((1, 4, 2), dtype=np.float32)
    share_obs = np.zeros((1, 4, 8), dtype=np.float32)
    actions = np.zeros((1, 4, 1), dtype=np.float32)
    rewards = np.zeros((1, 4, 1), dtype=np.float32)
    terminated = np.zeros((1, 4, 1), dtype=bool)
    terminated[0, 1, 0] = True
    truncated = np.zeros_like(terminated)
    infos = np.array([{}], dtype=object)
    log_probs = np.zeros((1, 4, 1), dtype=np.float32)
    values = np.zeros((1, 4, 1), dtype=np.float32)
    rnn_actor = np.ones((1, 4, 1, 3), dtype=np.float32)
    rnn_critic = np.ones_like(rnn_actor)

    runner.insert(
        (
            obs,
            share_obs,
            actions,
            rewards,
            terminated,
            truncated,
            infos,
            log_probs,
            values,
            rnn_actor,
            rnn_critic,
        )
    )

    inserted_masks = runner.buffer.args[4]
    inserted_actor_state = runner.buffer.args[7]
    inserted_critic_state = runner.buffer.args[8]
    active_masks = runner.buffer.kwargs["active_masks"]
    assert inserted_masks[0, 1, 0] == 0
    assert active_masks[0, 1, 0] == 0
    np.testing.assert_array_equal(inserted_actor_state[0, 1], 0)
    np.testing.assert_array_equal(inserted_critic_state[0, 1], 0)


class _PotentialReward(BaseRewardFunction):
    def get_reward(self, task, env, agent_id):
        raise NotImplementedError


def test_potential_reward_includes_discount_factor():
    config = SimpleNamespace(
        _PotentialReward_potential=True,
        _PotentialReward_scale=1.0,
        reward_gamma=0.9,
    )
    reward = _PotentialReward(config)
    reward.pre_rewards["agent"] = 2.0

    assert reward._process(3.0, "agent") == pytest.approx(0.7)
