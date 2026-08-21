import os
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))

from envs.env_wrappers import (
    ShareSubprocVecEnv,
    SubprocVecEnv,
    bind_current_process_to_rollout_cores,
    get_rollout_cpu_cores,
)
from config import get_config
from scripts.train.train_jsbsim import make_train_env, parse_args


class AffinityProbeEnv:
    observation_space = spaces.Box(low=-1, high=1, shape=(1,))
    action_space = spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def close(self):
        pass


class SharedAffinityProbeEnv(AffinityProbeEnv):
    share_observation_space = spaces.Box(low=-1, high=1, shape=(1,))

    def get_state(self):
        return np.zeros(1, dtype=np.float32)


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"),
    reason="CPU affinity is only available on Linux",
)
@pytest.mark.parametrize(
    "vec_env_class, env_class",
    [
        (SubprocVecEnv, AffinityProbeEnv),
        (ShareSubprocVecEnv, SharedAffinityProbeEnv),
    ],
)
def test_rollout_workers_are_pinned_to_matching_cpu_cores(vec_env_class, env_class):
    available_cores = tuple(sorted(os.sched_getaffinity(0)))
    worker_count = min(4, len(available_cores))

    cpu_cores = get_rollout_cpu_cores(worker_count)
    envs = vec_env_class([env_class for _ in cpu_cores], cpu_affinity=cpu_cores)
    try:
        envs.reset()
        assert envs.get_worker_cpu_affinities() == [(core,) for core in cpu_cores]
    finally:
        envs.close()


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"),
    reason="CPU affinity is only available on Linux",
)
def test_train_jsbsim_pins_twenty_workers_without_restricting_main_process():
    available_cpu_cores = tuple(sorted(os.sched_getaffinity(0)))
    if len(available_cpu_cores) < 20:
        pytest.skip("Fewer than twenty CPU cores are available to this process")
    expected_cpu_cores = available_cpu_cores[:20]

    original_affinity = os.sched_getaffinity(0)
    envs = None
    try:
        parser = get_config()
        all_args = parse_args([
            "--env-name", "SingleControl",
            "--scenario-name", "1/heading",
            "--n-rollout-threads", "20",
        ], parser)
        cpu_cores = get_rollout_cpu_cores(all_args.n_rollout_threads)
        assert cpu_cores == expected_cpu_cores
        assert os.sched_getaffinity(0) == original_affinity

        envs = make_train_env(all_args, cpu_affinity=cpu_cores)
        assert envs.get_worker_cpu_affinities() == [(core,) for core in expected_cpu_cores]
    finally:
        if envs is not None:
            envs.close()
        os.sched_setaffinity(0, original_affinity)
