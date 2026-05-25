from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from gg4_wk2.estimator import BaseEstimator
from gg4_wk2.simulation import Simulation
from gg4_wk2.simulation_library import SimulationLibrary


def _fit_alignment(X_hat: np.ndarray, X_true: np.ndarray) -> np.ndarray:
    T, _, _, _ = np.linalg.lstsq(X_hat, X_true, rcond=None)
    return T


def _r2(X_true: np.ndarray, X_hat: np.ndarray, T: np.ndarray) -> float:
    ss_res = float(np.sum((X_true - X_hat @ T) ** 2))
    ss_tot = float(np.sum((X_true - X_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


class Evaluator:
    def __init__(
        self,
        estimator: BaseEstimator,
        library: SimulationLibrary,
        *,
        train_frac: float = 0.5,
    ) -> None:
        self.estimator = estimator
        self.library = library
        self.train_frac = train_frac

    def _evaluate_simulation(self, simulation: Simulation) -> dict[str, float]:
        cond = simulation.condition
        T = cond.time_steps
        state_dim = cond.model.state_dim
        input_dim = cond.model.input_dim

        X_true = simulation.latent_states  # (n_trials, T, state_dim)
        U_true = cond.input.to_signal(T, input_dim).data  # (T, input_dim)
        Y = simulation.observations.observations  # (n_trials, T, n_neurons)

        X_hat, U_hat = self.estimator.estimate_latent_and_input(Y, state_dim, input_dim)

        if X_hat.ndim == 2:
            n_trials = X_true.shape[0]
            X_hat = np.tile(
                X_hat[np.newaxis], (n_trials, 1, 1)
            )  # (n_trials, T, latent_dim)

        if U_hat.ndim == 3:
            U_hat = U_hat.mean(axis=0)  # (T, input_dim)

        t_split = int(self.train_frac * T)

        X_true_train = X_true[:, :t_split, :].reshape(-1, state_dim)
        X_hat_train = X_hat[:, :t_split, :].reshape(-1, X_hat.shape[-1])
        T_x = _fit_alignment(X_hat_train, X_true_train)

        X_true_test = X_true[:, t_split:, :].reshape(-1, state_dim)
        X_hat_test = X_hat[:, t_split:, :].reshape(-1, X_hat.shape[-1])
        r2_x = _r2(X_true_test, X_hat_test, T_x)

        U_true_train = U_true[:t_split]
        U_hat_train = U_hat[:t_split]
        T_u = _fit_alignment(U_hat_train, U_true_train)

        U_true_test = U_true[t_split:]
        U_hat_test = U_hat[t_split:]
        r2_u = _r2(U_true_test, U_hat_test, T_u)

        return {"r2_x": r2_x, "r2_u": r2_u}

    def evaluate_library(self, *, show_progress: bool = True) -> pd.DataFrame:
        rows: list[dict[str, float]] = []

        with tqdm(
            total=len(self.library), desc="Evaluating", disable=not show_progress
        ) as pbar:
            for simulation in self.library:
                rows.append(self._evaluate_simulation(simulation))
                pbar.update(1)

        return pd.DataFrame(rows)
