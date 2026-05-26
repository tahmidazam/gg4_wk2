from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import subspace_angles
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


def _rmse(X_true: np.ndarray, X_hat: np.ndarray, T: np.ndarray) -> float:
    return float(np.sqrt(np.mean((X_true - X_hat @ T) ** 2)))


def _pearson_r(X_true: np.ndarray, X_hat: np.ndarray, T: np.ndarray) -> float:
    X_aligned = X_hat @ T  # (n_samples, dim)
    n_dims = X_true.shape[1]
    rs = []
    for d in range(n_dims):
        col_true = X_true[:, d]
        col_hat = X_aligned[:, d]
        std_true = col_true.std()
        std_hat = col_hat.std()
        if std_true < 1e-12 or std_hat < 1e-12:
            continue
        r = float(np.corrcoef(col_true, col_hat)[0, 1])
        rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def _mean_subspace_angle(X_true: np.ndarray, X_hat: np.ndarray) -> float:
    try:
        angles = subspace_angles(X_true, X_hat)
        return float(np.mean(angles))
    except np.linalg.LinAlgError:
        return float("nan")


def _param_r2(M_true: np.ndarray, M_hat: np.ndarray) -> float:
    flat_true = M_true.ravel()
    flat_hat = M_hat.ravel()
    ss_res = float(np.sum((flat_true - flat_hat) ** 2))
    ss_tot = float(np.sum((flat_true - flat_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


class Evaluator:
    def __init__(
        self,
        estimator: BaseEstimator,
        library: SimulationLibrary,
        *,
        train_frac: float = 0.5,
        n_jobs: int = -1,
    ) -> None:
        self.estimator = estimator
        self.library = library
        self.train_frac = train_frac
        self.n_jobs = n_jobs

    def _evaluate_simulation(self, simulation: Simulation) -> dict[str, float]:
        cond = simulation.condition
        T = cond.time_steps
        state_dim = cond.model.state_dim
        input_dim = cond.model.input_dim

        X_true = simulation.latent_states  # (n_trials, T, state_dim)
        U_true = cond.input.to_signal(T, input_dim).data  # (T, input_dim)
        Y = simulation.observations.observations  # (n_trials, T, n_neurons)

        X_hat, U_hat = self.estimator.estimate_latent_and_input(Y, state_dim, input_dim)

        # Snapshot fitted_params immediately after the call (thread safety: same thread)
        _raw_fp = getattr(self.estimator, "fitted_params", None)
        fitted_params: dict | None = _raw_fp if isinstance(_raw_fp, dict) else None

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

        result: dict[str, float] = {
            "r2_x": r2_x,
            "r2_u": r2_u,
            "rmse_x": _rmse(X_true_test, X_hat_test, T_x),
            "rmse_u": _rmse(U_true_test, U_hat_test, T_u),
            "pearson_r_x": _pearson_r(X_true_test, X_hat_test, T_x),
            "pearson_r_u": _pearson_r(U_true_test, U_hat_test, T_u),
            "mean_subspace_angle_x": _mean_subspace_angle(X_true_test, X_hat_test),
            "mean_subspace_angle_u": _mean_subspace_angle(U_true_test, U_hat_test),
        }

        if fitted_params is not None:
            A_hat = fitted_params["A"]
            B_hat = fitted_params["B"]
            C_hat = fitted_params["C"]
            R_hat = fitted_params["R"]

            # Align estimated parameters into the true latent basis.
            # T_x satisfies X_true ≈ X_hat @ T_x, i.e. x_hat ≈ x_true @ inv(T_x).
            # Row-vector dynamics: x_t = x_{t-1} A^T + u_{t-1} B^T
            # => A_aligned = T_x.T @ A_hat @ inv(T_x).T
            #    B_aligned = T_x.T @ B_hat @ T_u.T
            #    C_aligned = C_hat @ inv(T_x).T   (C is in observation space)
            try:
                T_x_inv = np.linalg.inv(T_x)
                A_aligned = T_x.T @ A_hat @ T_x_inv.T
                result["r2_A"] = _param_r2(cond.model.A, A_aligned)

                if input_dim > 0:
                    B_aligned = T_x.T @ B_hat @ T_u.T
                    result["r2_B"] = _param_r2(cond.model.B, B_aligned)

                C_aligned = C_hat @ T_x_inv.T
                result["r2_C"] = _param_r2(cond.model.C, C_aligned)

                result["r2_R"] = _param_r2(cond.model.R, R_hat)
            except np.linalg.LinAlgError:
                result["r2_A"] = float("nan")
                result["r2_B"] = float("nan")
                result["r2_C"] = float("nan")
                result["r2_R"] = float("nan")

        return result

    def evaluate_library(self, *, show_progress: bool = True) -> pd.DataFrame:
        simulations = list(self.library)

        rows = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._evaluate_simulation)(sim)
            for sim in tqdm(simulations, desc="Evaluating", disable=not show_progress)
        )

        return pd.DataFrame(rows)
