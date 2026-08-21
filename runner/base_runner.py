import os
import sys
import json
import platform
import random
import subprocess
from importlib import metadata
import wandb
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.buffer import ReplayBuffer
import logging

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.render_mode = config.get('render_mode', 'txt')
        
        # Tacview render obj
        self.tacview = None
        if self.render_mode == "real_time":
            from runner.tacview import Tacview
            self.tacview = Tacview()

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.buffer_size = self.all_args.buffer_size
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes
        self.selfplay_interval = self.all_args.selfplay_interval
        if self.selfplay_interval < 1:
            raise ValueError("--selfplay-interval must be at least 1")
        self.total_num_steps = 0
        self.start_episode = 0

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.save_dir = str(self.run_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.load()

    def load(self):
        # algorithm
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args,
                             self.envs.observation_space,
                             self.envs.action_space,
                             device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args,
                                   self.envs.observation_space,
                                   self.envs.action_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode=None):
        """Save inference weights and a resumable rollout-boundary checkpoint."""
        policy_actor = self.policy.actor
        policy_critic = self.policy.critic
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")

        checkpoint = {
            "format_version": 1,
            "actor": policy_actor.state_dict(),
            "critic": policy_critic.state_dict(),
            "optimizer": self.policy.optimizer.state_dict(),
            "episode": None if episode is None else int(episode),
            "total_num_steps": int(self.total_num_steps),
            "rng_state": self._capture_rng_state(),
            "config": dict(vars(self.all_args)),
            "runtime": self._runtime_metadata(),
            "extra": self._extra_checkpoint_state(),
        }
        checkpoint_path = os.path.join(self.save_dir, "training_state_latest.pt")
        temporary_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def restore(self):
        checkpoint_path = os.path.join(str(self.model_dir), "training_state_latest.pt")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self.policy.actor.load_state_dict(checkpoint["actor"])
            self.policy.critic.load_state_dict(checkpoint["critic"])
            self.policy.optimizer.load_state_dict(checkpoint["optimizer"])
            episode = checkpoint.get("episode")
            self.start_episode = 0 if episode is None else int(episode) + 1
            self.total_num_steps = int(checkpoint.get("total_num_steps", 0))
            self._restore_extra_checkpoint_state(checkpoint.get("extra", {}))
            self._restore_rng_state(checkpoint.get("rng_state", {}))
            logging.info(
                "Resumed training state from %s at update %s (%s steps)",
                checkpoint_path,
                self.start_episode,
                self.total_num_steps,
            )
            return

        policy_actor_state_dict = torch.load(
            str(self.model_dir) + '/actor_latest.pt',
            map_location=self.device,
            weights_only=True,
        )
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(
            str(self.model_dir) + '/critic_latest.pt',
            map_location=self.device,
            weights_only=True,
        )
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    @staticmethod
    def _capture_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _extra_checkpoint_state(self):
        extra = {}
        for name in (
            "policy_pool",
            "policy_snapshots",
            "latest_elo",
            "current_opponent_ids",
        ):
            if hasattr(self, name):
                extra[name] = getattr(self, name)
        return extra

    def _restore_extra_checkpoint_state(self, extra):
        for name, value in extra.items():
            setattr(self, name, value)

    @staticmethod
    def _runtime_metadata():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None

        packages = {}
        for package in ("torch", "gymnasium", "jsbsim", "numpy"):
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = None
        return {
            "git_commit": commit,
            "python": platform.python_version(),
            "packages": packages,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    def log_info(self, infos, total_num_steps):
        if self.use_wandb:
            for k, v in infos.items():
                wandb.log({k: v}, step=total_num_steps)
        else:
            record = {"total_num_steps": int(total_num_steps)}
            record.update({key: self._json_value(value) for key, value in infos.items()})
            metrics_path = os.path.join(self.save_dir, "metrics.jsonl")
            with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(record, allow_nan=False) + "\n")

    @staticmethod
    def _json_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.item()
            else:
                return value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def rollout_reward_metrics(self):
        """Return stable reward metrics even when no episode ends in a rollout."""
        completed = int(np.count_nonzero(self.buffer.masks[1:] == 0))
        metrics = {
            "rollout_average_step_reward": float(np.mean(self.buffer.rewards)),
            "completed_agent_episodes": completed,
        }
        if completed:
            metrics["average_episode_rewards"] = float(
                self.buffer.rewards.sum() / completed
            )
        return metrics
        
    def render_with_tacview(self, data):
        """
        Send data to Tacview for real-time rendering.
        :param data: The data to be rendered, which can be a string or a specific structure.
        """
        if self.tacview:
            try:
                self.tacview.send_data_to_client(data)
            except Exception as e:
                logging.error(f"Tacview rendering error: {e}")
