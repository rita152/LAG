import json
import random
from collections import deque
from types import SimpleNamespace

from gymnasium import spaces
import numpy as np
import torch

from envs.JSBSim.reward_functions import EventDrivenReward
from envs.JSBSim.tasks.multiplecombat_task import (
    HierarchicalMultipleCombatTask,
    MultipleCombatTask,
)
from envs.JSBSim.tasks.singlecombat_with_missle_task import (
    SingleCombatDodgeMissileTask,
)
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from algorithms.utils.selfplay import FSP, PFSP, SP
from runner.base_runner import Runner
from runner.jsbsim_runner import JSBSimRunner
from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner


class _InferenceProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_enabled = None

    def forward(self, observation, rnn_state):
        self.grad_enabled = torch.is_grad_enabled()
        return torch.zeros((1, 4)), torch.zeros((1, 1, 3))


def test_hierarchical_low_level_policy_uses_inference_mode():
    task = HierarchicalMultipleCombatTask.__new__(HierarchicalMultipleCombatTask)
    task.norm_delta_altitude = np.array([0.1, 0.0, -0.1])
    task.norm_delta_heading = np.array([-0.5, 0.0, 0.5])
    task.norm_delta_velocity = np.array([0.05, 0.0, -0.05])
    task._inner_rnn_states = {"A0100": np.zeros((1, 1, 3))}
    task.get_obs = lambda env, agent_id: np.zeros(9, dtype=np.float32)
    task.lowlevel_policy = _InferenceProbe()

    task.normalize_action(SimpleNamespace(), "A0100", np.array([1, 1, 1]))

    assert task.lowlevel_policy.grad_enabled is False


def test_local_metrics_are_persisted_as_json_lines(tmp_path):
    runner = Runner.__new__(Runner)
    runner.use_wandb = False
    runner.save_dir = str(tmp_path)

    runner.log_info(
        {
            "ratio": np.float32(1.0),
            "entropy": torch.tensor(0.25),
            "not_available": float("nan"),
        },
        total_num_steps=123,
    )

    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record == {
        "total_num_steps": 123,
        "ratio": 1.0,
        "entropy": 0.25,
        "not_available": None,
    }


def test_rollout_reward_metrics_do_not_divide_by_zero():
    runner = Runner.__new__(Runner)
    runner.buffer = SimpleNamespace(
        rewards=np.ones((4, 2, 1, 1), dtype=np.float32),
        masks=np.ones((5, 2, 1, 1), dtype=np.float32),
    )

    metrics = runner.rollout_reward_metrics()

    assert metrics["rollout_average_step_reward"] == 1.0
    assert metrics["completed_agent_episodes"] == 0
    assert metrics["completed_env_episodes"] == 0
    assert "average_episode_rewards" not in metrics


def test_episode_returns_span_rollouts_and_ignore_repeated_dead_masks():
    runner = Runner.__new__(Runner)
    runner.n_rollout_threads = 1
    runner.buffer = SimpleNamespace(
        num_agents=2,
        rewards=np.ones((4, 1, 2, 1), dtype=np.float32),
    )
    runner.reset_episode_metrics()

    def record(rewards, dones):
        rewards = np.asarray(rewards, dtype=np.float32).reshape(1, 2, 1)
        terminated = np.asarray(dones, dtype=bool).reshape(1, 2, 1)
        runner.record_episode_metrics(
            rewards, terminated, np.zeros_like(terminated)
        )

    record([1.0, 10.0], [False, False])
    record([2.0, 20.0], [True, False])
    record([0.0, 30.0], [True, False])

    first_rollout = runner.rollout_reward_metrics()
    assert first_rollout["completed_agent_episodes"] == 1
    assert first_rollout["average_episode_rewards"] == 3.0
    assert first_rollout["completed_env_episodes"] == 0

    record([0.0, 40.0], [True, True])
    second_rollout = runner.rollout_reward_metrics()
    assert second_rollout["completed_agent_episodes"] == 1
    assert second_rollout["average_episode_rewards"] == 100.0
    assert second_rollout["completed_env_episodes"] == 1
    assert second_rollout["average_team_episode_rewards"] == 51.5

    drained = runner.rollout_reward_metrics()
    assert drained["completed_agent_episodes"] == 0
    assert drained["completed_env_episodes"] == 0


def _checkpoint_runner(
    directory, actor_value, optimizer_lr, runner_class=Runner
):
    runner = runner_class.__new__(runner_class)
    actor = torch.nn.Linear(1, 1)
    critic = torch.nn.Linear(1, 1)
    with torch.no_grad():
        actor.weight.fill_(actor_value)
        critic.weight.fill_(-actor_value)
    policy = SimpleNamespace(
        actor=actor,
        critic=critic,
        optimizer=torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=optimizer_lr
        ),
    )
    runner.policy = policy
    runner.device = torch.device("cpu")
    runner.save_dir = str(directory)
    runner.model_dir = str(directory)
    runner.all_args = SimpleNamespace(seed=7, gamma=0.99)
    runner.total_num_steps = 456
    runner.start_episode = 0
    return runner


def test_ppo_runner_save_paths_write_resumable_checkpoints(tmp_path):
    for runner_class in (JSBSimRunner, SelfplayJSBSimRunner):
        save_dir = tmp_path / runner_class.__name__
        save_dir.mkdir()
        runner = _checkpoint_runner(
            save_dir,
            actor_value=2.0,
            optimizer_lr=0.123,
            runner_class=runner_class,
        )

        runner.save(episode=4)

        checkpoint = torch.load(
            save_dir / "training_state_latest.pt", weights_only=False
        )
        assert checkpoint["episode"] == 4
        assert "optimizer" in checkpoint
        assert (save_dir / "actor_latest.pt").is_file()
        assert (save_dir / "critic_latest.pt").is_file()


class _ConstantShapingReward:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def get_reward(self, task, env, agent_id):
        self.calls += 1
        return self.value


def test_dead_multiple_combat_agent_receives_only_terminal_event_reward():
    shaping = _ConstantShapingReward(50.0)
    event_reward = EventDrivenReward(SimpleNamespace())
    task = MultipleCombatTask.__new__(MultipleCombatTask)
    task.reward_functions = [shaping, event_reward]
    task._agent_die_flag = {}
    env = SimpleNamespace(
        agents={"A0100": SimpleNamespace(is_alive=False)},
        _events=[
            {
                "type": "aircraft_shotdown",
                "agent_id": "A0100",
                "processed": False,
            }
        ],
    )

    reward, _ = task.get_reward(env, "A0100", info={})
    env._events = [
        {
            "type": "missile_hit",
            "agent_id": "A0100",
            "processed": False,
        }
    ]
    repeated_reward, _ = task.get_reward(env, "A0100", info={})

    assert reward == -200.0
    assert repeated_reward == 0.0
    assert env._events[0]["processed"] is False
    assert shaping.calls == 0


class _DodgeAircraft:
    def __init__(self, alive, state=None):
        self.is_alive = alive
        self.enemies = []
        self._state = np.zeros(16) if state is None else np.asarray(state)

    def get_property_values(self, variables):
        return self._state

    def get_position(self):
        return np.zeros(3)

    def get_velocity(self):
        return np.array([200.0, 0.0, 0.0])

    def check_missile_warning(self):
        return None


def test_dodge_task_masks_and_never_launches_at_dead_target():
    target = _DodgeAircraft(alive=False)
    agent = _DodgeAircraft(alive=True)
    agent.enemies = [target]
    task = SingleCombatDodgeMissileTask.__new__(
        SingleCombatDodgeMissileTask
    )
    task.state_var = list(range(16))
    task.observation_space = spaces.Box(
        low=-10, high=10, shape=(task.base_obs_length,)
    )
    task.use_artillery = False
    task.max_attack_angle = 180.0
    task.max_attack_distance = np.inf
    task.min_attack_interval = 0
    task._last_shoot_time = {"A0100": 0}
    task.remaining_missiles = {"A0100": 1}
    task.lock_duration = {"A0100": deque([True], maxlen=1)}
    created_missiles = []
    env = SimpleNamespace(
        agents={"A0100": agent},
        center_lon=120.0,
        center_lat=60.0,
        center_alt=0.0,
        current_step=1,
        add_temp_simulator=created_missiles.append,
    )

    observation = task.get_obs(env, "A0100")
    task.step(env)

    assert observation.shape == (22,)
    np.testing.assert_array_equal(observation[9:15], np.zeros(6))
    assert observation[21] == 0.0
    assert created_missiles == []
    assert task.remaining_missiles["A0100"] == 1
    assert len(task.lock_duration["A0100"]) == 0


def test_checkpoint_restores_optimizer_rng_progress_and_selfplay_population(tmp_path):
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    runner = _checkpoint_runner(tmp_path, actor_value=2.0, optimizer_lr=0.123)
    runner.policy_pool = {"initial": 1000.0}
    runner.policy_snapshots = {"initial": runner.policy.actor.state_dict()}
    runner.latest_elo = 1012.0
    runner.save(episode=4)

    expected_random = (random.random(), np.random.rand(), torch.rand(1))
    restored = _checkpoint_runner(tmp_path, actor_value=-9.0, optimizer_lr=0.9)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored.restore()

    assert restored.start_episode == 5
    assert restored.total_num_steps == 456
    assert restored.policy.optimizer.param_groups[0]["lr"] == 0.123
    torch.testing.assert_close(restored.policy.actor.weight, torch.full((1, 1), 2.0))
    assert restored.policy_pool == {"initial": 1000.0}
    assert restored.latest_elo == 1012.0
    actual_random = (random.random(), np.random.rand(), torch.rand(1))
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    torch.testing.assert_close(actual_random[2], expected_random[2])

    checkpoint = torch.load(
        tmp_path / "training_state_latest.pt", weights_only=False
    )
    assert checkpoint["config"]["gamma"] == 0.99
    assert "git_commit" in checkpoint["runtime"]
    assert "torch" in checkpoint["runtime"]["packages"]


class _ActorOnlyPolicy:
    def __init__(self, value):
        self.actor = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.actor.weight.fill_(value)

    def prep_rollout(self):
        self.actor.eval()


def test_selfplay_starts_from_current_actor_and_refreshes_without_evaluation(tmp_path):
    runner = SelfplayJSBSimRunner.__new__(SelfplayJSBSimRunner)
    runner.policy = _ActorOnlyPolicy(3.0)
    runner.opponent_policy = [_ActorOnlyPolicy(-7.0)]
    runner.selfplay_algo = SP
    runner.policy_pool = {}
    runner.policy_snapshots = {}
    runner.current_opponent_ids = []
    runner.init_elo = 1000.0
    runner.latest_elo = 1000.0
    runner.save_dir = str(tmp_path)
    runner.model_dir = None
    runner.device = torch.device("cpu")

    runner._initialize_opponents()
    torch.testing.assert_close(
        runner.opponent_policy[0].actor.weight, runner.policy.actor.weight
    )

    with torch.no_grad():
        runner.policy.actor.weight.fill_(5.0)
    runner._store_policy_snapshot("1", runner.latest_elo)
    runner.reset_opponent(reset_environment=False)
    assert runner.current_opponent_ids == ["1"]
    torch.testing.assert_close(
        runner.opponent_policy[0].actor.weight, runner.policy.actor.weight
    )


def test_all_selfplay_algorithms_update_elo_from_results():
    for algorithm in (SP, FSP, PFSP):
        pool = {"opponent": 1000.0}
        new_ego_elo = algorithm.update(
            pool,
            {
                "ego_elo": 1000.0,
                "opponent_ids": ["opponent"],
                "actual_scores": [1.0],
            },
        )
        assert new_ego_elo == 1016.0
        assert pool["opponent"] == 984.0


def _initial_states():
    return {
        "A0100": {
            "ic_long_gc_deg": 120.0,
            "ic_lat_geod_deg": 60.0,
            "ic_h_sl_ft": 20000.0,
            "ic_psi_true_deg": 0.0,
            "ic_u_fps": 800.0,
        },
        "A0200": {
            "ic_long_gc_deg": 120.01,
            "ic_lat_geod_deg": 60.0,
            "ic_h_sl_ft": 20000.0,
            "ic_psi_true_deg": 0.0,
            "ic_u_fps": 800.0,
        },
        "B0100": {
            "ic_long_gc_deg": 120.0,
            "ic_lat_geod_deg": 60.1,
            "ic_h_sl_ft": 20000.0,
            "ic_psi_true_deg": 180.0,
            "ic_u_fps": 800.0,
        },
        "B0200": {
            "ic_long_gc_deg": 120.01,
            "ic_lat_geod_deg": 60.1,
            "ic_h_sl_ft": 20000.0,
            "ic_psi_true_deg": 180.0,
            "ic_u_fps": 800.0,
        },
    }


def _randomization_probe(seed):
    env = MultipleCombatEnv.__new__(MultipleCombatEnv)
    env.ego_ids = ["A0100", "A0200"]
    env.enm_ids = ["B0100", "B0200"]
    env.center_lon = 120.0
    env.center_lat = 60.0
    env.config = SimpleNamespace(
        initial_heading_jitter_deg=15.0,
        initial_altitude_jitter_ft=1000.0,
        initial_speed_jitter_fps=50.0,
    )
    env.np_random = np.random.default_rng(seed)
    return env


def test_multiple_combat_initial_state_randomization_is_seeded_and_mirrored():
    base = _initial_states()
    first = _randomization_probe(17)._randomize_initial_states(base)
    repeated = _randomization_probe(17)._randomize_initial_states(base)
    different = _randomization_probe(18)._randomize_initial_states(base)

    assert first == repeated
    assert first != different
    assert any(
        state["ic_h_sl_ft"] != 20000.0 for state in first.values()
    )

    env = _randomization_probe(1)
    mirrored = env._mirror_initial_states(base)
    for uid, state in base.items():
        assert mirrored[uid]["ic_long_gc_deg"] == 2 * env.center_lon - state["ic_long_gc_deg"]
        assert mirrored[uid]["ic_lat_geod_deg"] == 2 * env.center_lat - state["ic_lat_geod_deg"]
        assert mirrored[uid]["ic_psi_true_deg"] == (state["ic_psi_true_deg"] + 180.0) % 360.0
