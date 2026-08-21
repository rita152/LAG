from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)
        # Sparse transition events are already temporal differences. Applying
        # potential shaping would create a compensating reward on the next step.
        self.is_potential = False

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        for event in env._events:
            if event["processed"] or event["agent_id"] != agent_id:
                continue
            if event["type"] in ("aircraft_shotdown", "aircraft_crash"):
                reward -= 200
            elif event["type"] == "missile_hit":
                reward += 200
            event["processed"] = True
        return self._process(reward, agent_id)
