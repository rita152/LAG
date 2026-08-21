import os
from pathlib import Path

import pytest

from config import get_config
from envs import env_wrappers
from scripts.train import train_jsbsim


def test_jsbsim_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit):
        train_jsbsim.parse_args(["--clip-params", "0.3"], get_config())


def test_jsbsim_cli_accepts_the_documented_clip_parameter():
    args = train_jsbsim.parse_args(["--clip-param", "0.3"], get_config())
    assert args.clip_param == pytest.approx(0.3)


@pytest.mark.parametrize(
    "positive,negative,dest",
    [
        ("--use-recurrent-policy", "--no-recurrent-policy", "use_recurrent_policy"),
        ("--use-gae", "--no-gae", "use_gae"),
        ("--use-max-grad-norm", "--no-max-grad-norm", "use_max_grad_norm"),
    ],
)
def test_boolean_cli_options_have_literal_meaning(positive, negative, dest):
    parser = get_config()
    assert getattr(parser.parse_args([positive]), dest) is True
    assert getattr(parser.parse_args([negative]), dest) is False


def test_rollout_affinity_uses_noncontiguous_available_cpu_ids(monkeypatch):
    monkeypatch.setattr(env_wrappers.os, "sched_getaffinity", lambda pid: {8, 10, 12, 14})
    monkeypatch.setattr(env_wrappers.os, "sched_setaffinity", lambda pid, cores: None)

    assert env_wrappers.get_rollout_cpu_cores(3) == (8, 10, 12)


def test_rollout_affinity_rejects_more_workers_than_available_cpus(monkeypatch):
    monkeypatch.setattr(env_wrappers.os, "sched_getaffinity", lambda pid: {8, 10})
    monkeypatch.setattr(env_wrappers.os, "sched_setaffinity", lambda pid, cores: None)

    with pytest.raises(ValueError, match="available"):
        env_wrappers.get_rollout_cpu_cores(3)


class _RunnerThatFails:
    def run(self):
        raise RuntimeError("training failed")


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_training_failure_is_rethrown_after_environment_cleanup():
    envs = _Closable()
    eval_envs = _Closable()
    assert hasattr(train_jsbsim, "run_and_close")

    with pytest.raises(RuntimeError, match="training failed"):
        train_jsbsim.run_and_close(_RunnerThatFails(), envs, eval_envs)

    assert envs.closed
    assert eval_envs.closed


def test_repository_declares_a_conda_environment():
    environment_file = Path(__file__).resolve().parents[1] / "environment.yml"
    assert environment_file.is_file()
