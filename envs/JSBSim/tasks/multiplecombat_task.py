import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class MultipleCombatTask(SingleCombatTask):
    relative_feature_size = 7

    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 4

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.obs_length = 9 + (self.num_agents - 1) * self.relative_feature_size
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 9
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            if not sim.is_alive:
                offset += self.relative_feature_size
                continue
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset] = 1.0
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += self.relative_feature_size
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if self._agent_die_flag.get(agent_id, False):
            return 0.0, info

        reward = 0.0
        agent_is_alive = env.agents[agent_id].is_alive
        self._agent_die_flag[agent_id] = not agent_is_alive
        for reward_function in self.reward_functions:
            if not agent_is_alive and not isinstance(
                reward_function, EventDrivenReward
            ):
                continue
            reward += reward_function.get_reward(self, env, agent_id)
        return reward, info


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    
    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(
            get_root_dir() + '/model/baseline_model.pt',
            map_location=torch.device('cpu'),
            weights_only=True,
        ))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        with torch.inference_mode():
            _action, _rnn_states = self.lowlevel_policy(
                input_obs, self._inner_rnn_states[agent_id]
            )
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)



class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    shoot_state_size = 4

    def __init__(self, config: str):
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
        # ego + (partners/enemies) + one missile-warning slot + shoot state
        self.obs_length = (
            9
            + self.num_agents * self.relative_feature_size
            + self.shoot_state_size
        )
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))
    
    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])


    def get_obs(self, env, agent_id):
        norm_obs = MultipleCombatTask.get_obs(self, env, agent_id)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        offset = 9 + (self.num_agents - 1) * self.relative_feature_size

        # one missile-warning slot
        missile_sim = env.agents[agent_id].check_missile_warning() #
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset] = 1.0
            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag

        shoot_offset = offset + self.relative_feature_size
        agent = env.agents[agent_id]
        target, _, _, shoot_valid = self._get_shoot_target(env, agent_id)
        max_missiles = max(agent.num_missiles, 1)
        shoot_interval = env.current_step - self._last_shoot_time[agent_id]
        cooldown = max(self.min_attack_interval - shoot_interval, 0)
        norm_obs[shoot_offset:] = (
            self._remaining_missiles[agent_id] / max_missiles,
            cooldown / max(self.min_attack_interval, 1),
            float(shoot_valid),
            float(target is not None),
        )
        return np.clip(norm_obs, self.observation_space.low, self.observation_space.high)

    def _get_shoot_target(self, env, agent_id):
        agent = env.agents[agent_id]
        alive_enemies = [enemy for enemy in agent.enemies if enemy.is_alive]
        if not agent.is_alive or not alive_enemies:
            return None, np.inf, 180.0, False

        offsets = [enemy.get_position() - agent.get_position() for enemy in alive_enemies]
        distances = [np.linalg.norm(offset) for offset in offsets]
        target_index = int(np.argmin(distances))
        target = alive_enemies[target_index]
        offset = offsets[target_index]
        distance = distances[target_index]
        heading = agent.get_velocity()
        attack_angle = np.rad2deg(np.arccos(np.clip(
            np.sum(offset * heading) / (distance * np.linalg.norm(heading) + 1e-8),
            -1,
            1,
        )))
        shoot_interval = env.current_step - self._last_shoot_time[agent_id]
        shoot_valid = (
            self._remaining_missiles[agent_id] > 0
            and attack_angle <= self.max_attack_angle
            and distance <= self.max_attack_distance
            and shoot_interval >= self.min_attack_interval
        )
        return target, distance, attack_angle, shoot_valid

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[3] > 0
        return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            target, _, _, shoot_valid = self._get_shoot_target(env, agent_id)
            shoot_flag = bool(self._shoot_action[agent_id] and shoot_valid)
            if shoot_flag:
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=target, uid=new_missile_uid))
                self._remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step
