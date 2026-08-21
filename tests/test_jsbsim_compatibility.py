from types import SimpleNamespace

import numpy as np
import pytest

from envs.JSBSim.core.catalog import MixedCatalog
from envs.JSBSim.core.simulatior import MissileSimulator
from envs.JSBSim.envs.env_base import BaseEnv
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.reward_functions.posture_reward import PostureReward


def test_catalog_accepts_multiline_string_from_current_jsbsim():
    catalog = MixedCatalog()
    catalog.add_jsbsim_props("audit/property-one (RW)\naudit/property-two (R)\n")

    assert catalog.audit_property_one.name_jsbsim == "audit/property-one"
    assert catalog.audit_property_one.access == "RW"
    assert catalog.audit_property_two.name_jsbsim == "audit/property-two"
    assert catalog.audit_property_two.access == "R"


def test_catalog_keeps_compatibility_with_iterable_property_lines():
    catalog = MixedCatalog()
    catalog.add_jsbsim_props(["audit/list-property (RW)"])
    assert catalog.audit_list_property.name_jsbsim == "audit/list-property"


def test_environment_pack_rejects_inf_without_entering_debugger():
    env = BaseEnv.__new__(BaseEnv)
    env.ego_ids = ["A0100"]
    env.enm_ids = []
    env.current_step = 3

    with pytest.raises(FloatingPointError, match="Non-finite"):
        env._pack({"A0100": np.array([np.inf], dtype=np.float32)})


class _KinematicTarget:
    def __init__(self, position, velocity, alive=True):
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self.is_alive = alive

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity


def test_missile_guidance_and_transition_stay_finite_at_singular_geometry():
    missile = MissileSimulator(uid="M0001", dt=1.0 / 60.0)
    missile.target_aircraft = _KinematicTarget([0, 0, 0], [0, 0, 0])
    missile._position[:] = 0
    missile._velocity[:] = 0
    missile._posture[:] = [0, np.pi / 2, 0]
    missile._t = 0.0
    missile._m = missile._m0
    missile._dtheta = 0.0
    missile._dphi = 0.0
    missile.lon0, missile.lat0, missile.alt0 = 120.0, 60.0, 0.0

    action, distance = missile._guidance()
    missile._state_trans(action)

    assert np.all(np.isfinite(action))
    assert np.isfinite(distance)
    assert np.all(np.isfinite(missile.get_position()))
    assert np.all(np.isfinite(missile.get_velocity()))


def test_posture_reward_ignores_dead_targets():
    ego = _KinematicTarget([0, 0, 0], [200, 0, 0])
    dead_enemy = _KinematicTarget([1000, 0, 0], [-200, 0, 0], alive=False)
    ego.enemies = [dead_enemy]
    env = SimpleNamespace(agents={"A0100": ego})
    reward = PostureReward(SimpleNamespace())

    assert reward.get_reward(None, env, "A0100") == 0.0


def test_multiple_combat_observation_masks_dead_slots_and_exposes_shoot_state():
    env = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay")
    try:
        env.seed(11)
        env.reset()
        agent_id = env.ego_ids[0]
        agent = env.agents[agent_id]
        dead_target = agent.enemies[0]
        target_slot = 1 + agent.enemies.index(dead_target)
        dead_target.crash()

        obs = env.task.get_obs(env, agent_id)
        start = 9 + target_slot * env.task.relative_feature_size
        np.testing.assert_array_equal(
            obs[start:start + env.task.relative_feature_size], 0.0
        )
        assert obs.shape == env.observation_space.shape
        assert obs[-4] == 1.0  # all missiles remain
        assert obs[-1] == 1.0  # another live target exists
    finally:
        env.close()


def test_multiple_combat_shooting_never_targets_a_dead_aircraft():
    env = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay")
    try:
        env.seed(13)
        env.reset()
        agent_id = env.ego_ids[0]
        agent = env.agents[agent_id]
        dead_target = min(
            agent.enemies,
            key=lambda enemy: np.linalg.norm(enemy.get_position() - agent.get_position()),
        )
        dead_target.crash()
        env.task._shoot_action = {
            uid: uid == agent_id for uid in env.agents.keys()
        }

        env.task.step(env)

        assert agent.launch_missiles
        assert agent.launch_missiles[-1].target_aircraft is not dead_target
        assert agent.launch_missiles[-1].target_aircraft.is_alive
    finally:
        env.close()
