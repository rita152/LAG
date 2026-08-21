import numpy as np
from typing import Dict, List
from abc import ABC, abstractstaticmethod


def update_elo_ratings(ego_elo, agents_elo, opponent_ids, actual_scores, k=32.0):
    """Update an ego rating and its evaluated historical opponents in place."""
    if len(opponent_ids) != len(actual_scores):
        raise ValueError("opponent_ids and actual_scores must have the same length")
    if not opponent_ids:
        return float(ego_elo)

    ego_updates = []
    grouped_scores = {}
    for opponent_id, actual_score in zip(opponent_ids, actual_scores):
        if opponent_id not in agents_elo:
            raise KeyError(f"Unknown self-play opponent {opponent_id!r}")
        score = float(actual_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Elo actual score must be in [0, 1], got {score}")
        grouped_scores.setdefault(opponent_id, []).append(score)

    for opponent_id, scores in grouped_scores.items():
        opponent_elo = float(agents_elo[opponent_id])
        actual_score = float(np.mean(scores))
        expected_score = 1.0 / (
            1.0 + 10.0 ** ((opponent_elo - float(ego_elo)) / 400.0)
        )
        delta = float(k) * (actual_score - expected_score)
        agents_elo[opponent_id] = opponent_elo - delta
        ego_updates.append(delta)

    return float(ego_elo) + float(np.mean(ego_updates))


def _update_from_results(agents_elo, eval_results, **kwargs):
    return update_elo_ratings(
        eval_results["ego_elo"],
        agents_elo,
        eval_results["opponent_ids"],
        eval_results["actual_scores"],
        k=kwargs.get("k", eval_results.get("k", 32.0)),
    )


def get_algorithm(algo_name):
    if algo_name == 'sp':
        return SP
    elif algo_name == 'fsp':
        return FSP
    elif algo_name == 'pfsp':
        return PFSP
    else:
        raise NotImplementedError("Unknown algorithm {}".format(algo_name))


class SelfplayAlgorithm(ABC):

    @abstractstaticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        pass

    @abstractstaticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class SP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return list(agents_elo.keys())[-1]

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        return _update_from_results(agents_elo, eval_results, **kwargs)


class FSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return np.random.choice(list(agents_elo.keys()))

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        return _update_from_results(agents_elo, eval_results, **kwargs)


class PFSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], lam=1, s=100, **kwargs) -> str:
        history_elo = np.array(list(agents_elo.values()))
        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
        """ meta-solver """
        k = float(len(sample_probs) + 1)
        meta_solver_probs = np.exp(lam / k * sample_probs) / np.sum(np.exp(lam / k * sample_probs))
        opponent_idx = np.random.choice(a=list(agents_elo.keys()), size=1, p=meta_solver_probs).item()
        return opponent_idx

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]]) -> None:
        return _update_from_results(agents_elo, eval_results)
