import numpy as np
import copy
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False
        self.training_mode = True
        self._evaluation_reset_index = 0
        self._base_init_states = {
            uid: copy.deepcopy(sim.init_state) for uid, sim in self.agents.items()
        }
        self._base_partner_ids = {
            uid: tuple(partner.uid for partner in sim.partners)
            for uid, sim in self.agents.items()
        }
        self._base_enemy_ids = {
            uid: tuple(enemy.uid for enemy in sim.enemies)
            for uid, sim in self.agents.items()
        }

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self, *, seed=None, options=None):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.reset_simulators(options=options)
        self._events = []
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs), {}

    def set_training_mode(self, training: bool):
        self.training_mode = bool(training)
        self._evaluation_reset_index = 0

    def seed(self, seed=None):
        result = super().seed(seed)
        self._evaluation_reset_index = 0
        return result

    def reset_simulators(self, options=None):
        states = {
            uid: copy.deepcopy(state)
            for uid, state in self._base_init_states.items()
        }
        randomize = bool(
            getattr(self.config, "randomize_initial_state", True)
        )
        if self.training_mode and randomize:
            states = self._randomize_initial_states(states)
            self._randomize_observation_slots()
        else:
            variant = (options or {}).get("initial_state_variant")
            if variant is None:
                variant = "base" if self._evaluation_reset_index % 2 == 0 else "mirror"
            if variant not in ("base", "mirror"):
                raise ValueError(
                    "initial_state_variant must be either 'base' or 'mirror'"
                )
            if variant == "mirror":
                states = self._mirror_initial_states(states)
            self._evaluation_reset_index += 1
            self._restore_observation_slots()

        for uid, sim in self._jsbsims.items():
            sim.reload(states[uid])
        self._tempsims.clear()

    def _randomize_initial_states(self, states):
        ego_states = [states[uid] for uid in self.ego_ids]
        enemy_states = [states[uid] for uid in self.enm_ids]
        self.np_random.shuffle(ego_states)
        self.np_random.shuffle(enemy_states)
        if len(ego_states) == len(enemy_states) and self.np_random.random() < 0.5:
            ego_states, enemy_states = enemy_states, ego_states
        states = {
            **{uid: copy.deepcopy(state) for uid, state in zip(self.ego_ids, ego_states)},
            **{uid: copy.deepcopy(state) for uid, state in zip(self.enm_ids, enemy_states)},
        }

        if self.np_random.random() < 0.5:
            states = self._mirror_initial_states(states)

        heading_jitter = float(
            getattr(self.config, "initial_heading_jitter_deg", 15.0)
        )
        altitude_jitter = float(
            getattr(self.config, "initial_altitude_jitter_ft", 1000.0)
        )
        speed_jitter = float(
            getattr(self.config, "initial_speed_jitter_fps", 50.0)
        )
        for state in states.values():
            state["ic_psi_true_deg"] = (
                state.get("ic_psi_true_deg", 0.0)
                + self.np_random.uniform(-heading_jitter, heading_jitter)
            ) % 360.0
            state["ic_h_sl_ft"] = state.get("ic_h_sl_ft", 20000.0) + self.np_random.uniform(
                -altitude_jitter, altitude_jitter
            )
            state["ic_u_fps"] = state.get("ic_u_fps", 800.0) + self.np_random.uniform(
                -speed_jitter, speed_jitter
            )
        return states

    def _mirror_initial_states(self, states):
        mirrored = copy.deepcopy(states)
        for state in mirrored.values():
            state["ic_long_gc_deg"] = (
                2 * self.center_lon - state.get("ic_long_gc_deg", self.center_lon)
            )
            state["ic_lat_geod_deg"] = (
                2 * self.center_lat - state.get("ic_lat_geod_deg", self.center_lat)
            )
            state["ic_psi_true_deg"] = (
                state.get("ic_psi_true_deg", 0.0) + 180.0
            ) % 360.0
        return mirrored

    def _restore_observation_slots(self):
        for uid, sim in self.agents.items():
            sim.partners = [self.agents[key] for key in self._base_partner_ids[uid]]
            sim.enemies = [self.agents[key] for key in self._base_enemy_ids[uid]]

    def _randomize_observation_slots(self):
        self._restore_observation_slots()
        for sim in self.agents.values():
            self.np_random.shuffle(sim.partners)
            self.np_random.shuffle(sim.enemies)

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        self._events = []

        # apply actions
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in list(self._tempsims.values()):
                sim.run()
        self.task.step(self)

        terminated = {}
        truncated = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            agent_truncated = bool(info.get("truncated", {}).get(agent_id, False))
            terminated[agent_id] = [bool(done and not agent_truncated)]
            truncated[agent_id] = [agent_truncated]

        self._collect_events()
        obs = self.get_obs()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        for ego_id in self.ego_ids:
            if self.agents[ego_id].is_alive:
                rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            if self.agents[enm_id].is_alive:
                rewards[enm_id] = [enm_reward]

        info["events"] = tuple(dict(event) for event in self._events)
        self._prune_temp_simulators()

        return (
            self._pack(obs),
            self._pack(rewards),
            self._pack(terminated),
            self._pack(truncated),
            info,
        )
