from types import SimpleNamespace

import numpy as np

from envs.JSBSim.core.simulatior import AircraftSimulator, MissileSimulator
from envs.JSBSim.envs.env_base import BaseEnv
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.reward_functions.event_driven_reward import EventDrivenReward


class _AircraftStub:
    def __init__(self, uid, position, velocity, alive=True):
        self.uid = uid
        self.color = "Red"
        self.dt = 1.0 / 60.0
        self.lon0 = 120.0
        self.lat0 = 60.0
        self.alt0 = 0.0
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._geodetic = np.array([120.0, 60.0, 5000.0], dtype=np.float64)
        self._rpy = np.zeros(3, dtype=np.float64)
        self._alive = alive
        self._shotdown = False
        self._crash = False
        self.death_event_reported = False
        self.launch_missiles = []
        self.under_missiles = []

    @property
    def is_alive(self):
        return self._alive

    @property
    def is_shotdown(self):
        return self._shotdown

    @property
    def is_crash(self):
        return self._crash

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def get_geodetic(self):
        return self._geodetic

    def get_rpy(self):
        return self._rpy

    def shotdown(self):
        if self._alive:
            self._alive = False
            self._shotdown = True


def _hit_missile():
    shooter = _AircraftStub("A0100", [0, 0, 5000], [300, 0, 0])
    target = _AircraftStub("B0100", [1, 0, 5000], [300, 0, 0])
    missile = MissileSimulator.create(shooter, target, "M0001")
    missile.run()
    return shooter, target, missile


def test_hit_and_miss_are_absorbing_missile_states():
    _, target, missile = _hit_missile()
    assert missile.is_success
    assert target.is_shotdown

    for _ in range(10):
        missile.run()

    assert missile.is_success
    assert missile.is_done


def test_combat_events_are_emitted_and_rewarded_exactly_once():
    shooter, target, missile = _hit_missile()
    env = BaseEnv.__new__(BaseEnv)
    env._jsbsims = {shooter.uid: shooter, target.uid: target}
    env._tempsims = {missile.uid: missile}
    env._events = []
    reward = EventDrivenReward(SimpleNamespace(EventDrivenReward_potential=True))

    env._collect_events()
    assert reward.get_reward(None, env, shooter.uid) == 200
    assert reward.get_reward(None, env, target.uid) == -200
    assert reward.get_reward(None, env, shooter.uid) == 0
    assert reward.get_reward(None, env, target.uid) == 0

    env._events = []
    env._collect_events()
    assert reward.get_reward(None, env, shooter.uid) == 0
    assert reward.get_reward(None, env, target.uid) == 0


def test_finished_missile_is_removed_after_render_grace_period():
    shooter, target, missile = _hit_missile()
    env = BaseEnv.__new__(BaseEnv)
    env._jsbsims = {shooter.uid: shooter, target.uid: target}
    env._tempsims = {missile.uid: missile}
    missile._left_t = 1
    missile.run()

    env._prune_temp_simulators()

    assert missile.uid not in env._tempsims
    assert missile not in shooter.launch_missiles
    assert missile not in target.under_missiles


def test_aircraft_terminal_cause_is_absorbing():
    aircraft = AircraftSimulator.__new__(AircraftSimulator)
    aircraft._BaseSimulator__uid = "A0100"
    aircraft._AircraftSimulator__status = AircraftSimulator.ALIVE
    aircraft.death_event_reported = False

    aircraft.shotdown()
    aircraft.crash()

    assert aircraft.is_shotdown
    assert not aircraft.is_crash


def test_multiple_combat_pays_terminal_penalty_on_death_transition():
    env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
    try:
        env.seed(7)
        env.reset()
        victim_id = env.ego_ids[1]
        victim_index = (env.ego_ids + env.enm_ids).index(victim_id)
        env.agents[victim_id].crash()
        actions = np.stack([env.action_space.sample() for _ in range(env.num_agents)])

        _, rewards, _, _, _ = env.step(actions)

        assert rewards[victim_index, 0] < -100
    finally:
        env.close()
