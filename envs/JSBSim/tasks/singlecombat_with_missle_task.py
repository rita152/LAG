import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    kinematic_obs_length = 21
    base_obs_length = kinematic_obs_length + 1

    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(
            low=-10, high=10., shape=(self.base_obs_length,)
        )

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        - target status
            - [21] enemy alive flag
        """
        norm_obs = np.zeros(self.base_obs_length)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        target_aircraft = env.agents[agent_id].enemies[0]
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enemy info. A dead target has an explicit zero slot
        # instead of a frozen kinematic state.
        if target_aircraft.is_alive:
            enm_obs_list = np.array(
                target_aircraft.get_property_values(self.state_var)
            )
            enm_cur_ned = LLA2NEU(
                *enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt
            )
            enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(
                ego_feature, enm_feature, return_side=True
            )
            norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
            norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
            norm_obs[11] = ego_AO
            norm_obs[12] = ego_TA
            norm_obs[13] = R / 10000
            norm_obs[14] = side_flag
            norm_obs[21] = 1.0
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            target_aircraft = agent.enemies[0]
            if not agent.is_alive or not target_aircraft.is_alive:
                self.lock_duration[agent_id].clear()
                continue
            target = target_aircraft.get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=target_aircraft, uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatDodgeMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class SingleCombatShootMissileTask(SingleCombatDodgeMissileTask):
    shoot_state_size = 4

    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(
            low=-10,
            high=10.,
            shape=(self.kinematic_obs_length + self.shoot_state_size,),
        )

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
    
    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.observation_space.shape[0])
        norm_obs[:self.kinematic_obs_length] = super().get_obs(
            env, agent_id
        )[:self.kinematic_obs_length]
        agent = env.agents[agent_id]
        target, shoot_valid = self._get_shoot_target(env, agent_id)
        max_missiles = max(agent.num_missiles, 1)
        shoot_interval = env.current_step - self._last_shoot_time[agent_id]
        cooldown = max(self.min_attack_interval - shoot_interval, 0)
        norm_obs[-self.shoot_state_size:] = (
            self.remaining_missiles[agent_id] / max_missiles,
            cooldown / max(self.min_attack_interval, 1),
            float(shoot_valid),
            float(target is not None),
        )
        return norm_obs

    def _get_shoot_target(self, env, agent_id):
        agent = env.agents[agent_id]
        alive_enemies = [enemy for enemy in agent.enemies if enemy.is_alive]
        if not agent.is_alive or not alive_enemies:
            return None, False
        target = min(
            alive_enemies,
            key=lambda enemy: np.linalg.norm(enemy.get_position() - agent.get_position()),
        )
        offset = target.get_position() - agent.get_position()
        distance = np.linalg.norm(offset)
        heading = agent.get_velocity()
        attack_angle = np.rad2deg(np.arccos(np.clip(
            np.sum(offset * heading) / (distance * np.linalg.norm(heading) + 1e-8),
            -1,
            1,
        )))
        shoot_interval = env.current_step - self._last_shoot_time[agent_id]
        valid = (
            self.remaining_missiles[agent_id] > 0
            and shoot_interval >= self.min_attack_interval
            and attack_angle <= self.max_attack_angle
            and distance <= self.max_attack_distance
        )
        return target, valid
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            target, shoot_valid = self._get_shoot_target(env, agent_id)
            shoot_flag = bool(self._shoot_action[agent_id] and shoot_valid)
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=target, uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step


class HierarchicalSingleCombatShootTask(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)
