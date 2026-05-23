from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from gg4_wk2.dataset import Dataset
from gg4_wk2.simulator import LinearGaussianModel, LinearGaussianSimulator
from gg4_wk2.waveform import Waveform


def _effective_rank(eigs: NDArray[np.float64]) -> np.float64:
    pos = eigs[eigs > 0]
    if pos.size == 0:
        return np.float64(0.0)
    p = pos / pos.sum()
    return np.exp(-np.sum(p * np.log(p)))


@dataclass(frozen=True)
class Condition:
    model: LinearGaussianModel
    x_0: Waveform
    input: Waveform
    time_steps: int
    trials: int

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "spectral_radius_A",
        "spectral_gap_A",
        "dominant_mode_time_constant",
        "dominant_mode_damping_ratio",
        "dominant_mode_oscillation_frequency",
        "henrici_non_normality_index",
        "controllability_min_eigenvalue",
        "controllability_trace",
        "controllability_log_det",
        "controllability_condition_number",
        "controllability_effective_rank",
        "observability_min_eigenvalue",
        "observability_trace",
        "observability_log_det",
        "observability_condition_number",
        "observability_effective_rank",
        "hankel_norm",
        "hankel_min_singular_value",
        "hankel_condition_number",
        "hankel_effective_rank",
        "process_noise_trace",
        "process_noise_log_det",
        "process_noise_condition_number",
        "measurement_noise_trace",
        "measurement_noise_log_det",
        "measurement_noise_condition_number",
        "steady_state_signal_to_noise_ratio",
        "filter_closed_loop_spectral_radius",
        "filter_covariance_trace",
        "filter_covariance_log_det",
        "innovation_covariance_trace",
        "log10_time_steps",
        "trials",
        "log10_x0_norm",
        "log10_input_rms",
        "log10_excitation_to_process_noise_ratio",
        "input_kind_id",
    )

    def run(self, *, seed: int | None = None) -> tuple[NDArray[np.float64], Dataset]:
        x_0 = self.x_0.to_signal(1, self.model.state_dim).data[0]
        u = self.input.to_signal(self.time_steps, self.model.input_dim).data
        return LinearGaussianSimulator(self.model, seed=seed).run(
            x_0, u, self.time_steps, self.trials
        )

    @property
    def features(self) -> dict[str, float]:
        A, B, C, Q, R = (
            self.model.A,
            self.model.B,
            self.model.C,
            self.model.Q,
            self.model.R,
        )

        eigs_A = np.linalg.eigvals(A)
        moduli_A = np.abs(eigs_A)
        spectral_radius = float(np.max(moduli_A))
        sorted_moduli = np.sort(moduli_A)[::-1]

        spectral_gap = (
            float(sorted_moduli[0] - sorted_moduli[1])
            if sorted_moduli.size >= 2
            else float(np.inf)
        )

        if spectral_radius >= 1.0:
            dominant_tc = float(np.inf)
        elif spectral_radius == 0.0:
            dominant_tc = 0.0
        else:
            dominant_tc = float(-1.0 / np.log(spectral_radius))

        dom_lam = eigs_A[np.argmax(moduli_A)]
        if dom_lam == 0:
            damping_ratio, oscillation_freq = 1.0, 0.0
        else:
            s = np.log(dom_lam)
            abs_s = float(np.abs(s))
            damping_ratio = float(-np.real(s) / abs_s) if abs_s != 0 else 1.0
            oscillation_freq = float(np.abs(np.imag(s)))

        frob_sq = float(np.sum(A**2))
        eig_energy = float(np.sum(moduli_A**2))
        henrici = (
            float(max(0.0, (frob_sq - eig_energy) / frob_sq)) if frob_sq != 0 else 0.0
        )

        if spectral_radius >= 1.0:
            raise ValueError(
                f"Gramians undefined: A is not Schur stable (spectral radius = {spectral_radius:.4f})."
            )
        W_c = solve_discrete_lyapunov(A, B @ B.T)
        W_o = solve_discrete_lyapunov(A.T, C.T @ C)

        ctrl_eigs = np.linalg.eigvalsh(W_c).astype(np.float64)
        ctrl_min = float(ctrl_eigs.min())
        ctrl_trace = float(np.trace(W_c))
        s_c, ldet = np.linalg.slogdet(W_c)
        ctrl_log_det = float(ldet) if s_c > 0 else float(-np.inf)
        ctrl_cond = (
            float(ctrl_eigs.max() / ctrl_eigs.min()) if ctrl_min > 0 else float(np.inf)
        )
        ctrl_eff_rank = float(_effective_rank(ctrl_eigs))

        obs_eigs = np.linalg.eigvalsh(W_o).astype(np.float64)
        obs_min = float(obs_eigs.min())
        obs_trace = float(np.trace(W_o))
        s_o, ldet = np.linalg.slogdet(W_o)
        obs_log_det = float(ldet) if s_o > 0 else float(-np.inf)
        obs_cond = (
            float(obs_eigs.max() / obs_eigs.min()) if obs_min > 0 else float(np.inf)
        )
        obs_eff_rank = float(_effective_rank(obs_eigs))

        hankel_eigs = np.linalg.eigvals(W_c @ W_o).real
        hsv = np.sort(np.sqrt(np.clip(hankel_eigs, 0.0, None)))[::-1]
        hankel_norm = float(hsv[0])
        hankel_min_sv = float(hsv[-1])
        hankel_cond = float(hsv[0] / hsv[-1]) if hsv[-1] > 0 else float(np.inf)
        hankel_eff_rank = float(_effective_rank(hsv))

        Q_trace = float(np.trace(Q))
        s_q, ldet = np.linalg.slogdet(Q)
        Q_log_det = float(ldet) if s_q > 0 else float(-np.inf)
        Q_eigs = np.linalg.eigvalsh(Q)
        Q_cond = (
            float(Q_eigs.max() / Q_eigs.min()) if Q_eigs.min() > 0 else float(np.inf)
        )

        R_trace = float(np.trace(R))
        s_r, ldet = np.linalg.slogdet(R)
        R_log_det = float(ldet) if s_r > 0 else float(-np.inf)
        R_eigs = np.linalg.eigvalsh(R)
        R_cond = (
            float(R_eigs.max() / R_eigs.min()) if R_eigs.min() > 0 else float(np.inf)
        )

        P_ss = solve_discrete_lyapunov(A, Q)
        signal_var = float(np.trace(C @ P_ss @ C.T))
        ss_snr = float(signal_var / R_trace) if R_trace != 0 else float(np.inf)

        P_kf = solve_discrete_are(A.T, C.T, Q, R)
        S_innov = C @ P_kf @ C.T + R
        K = A @ P_kf @ C.T @ np.linalg.inv(S_innov)
        filter_sr = float(np.max(np.abs(np.linalg.eigvals(A - K @ C))))
        P_kf_trace = float(np.trace(P_kf))
        s_p, ldet = np.linalg.slogdet(P_kf)
        P_kf_log_det = float(ldet) if s_p > 0 else float(-np.inf)
        innov_cov_trace = float(np.trace(S_innov))

        x0_norm = float(
            np.linalg.norm(self.x_0.to_signal(1, self.model.state_dim).data[0])
        )
        rms = self.input.root_mean_sqare(self.time_steps, self.model.input_dim)
        B_frob_sq = float(np.sum(B**2))

        return {
            "spectral_radius_A": spectral_radius,
            "spectral_gap_A": spectral_gap,
            "dominant_mode_time_constant": dominant_tc,
            "dominant_mode_damping_ratio": damping_ratio,
            "dominant_mode_oscillation_frequency": oscillation_freq,
            "henrici_non_normality_index": henrici,
            "controllability_min_eigenvalue": ctrl_min,
            "controllability_trace": ctrl_trace,
            "controllability_log_det": ctrl_log_det,
            "controllability_condition_number": ctrl_cond,
            "controllability_effective_rank": ctrl_eff_rank,
            "observability_min_eigenvalue": obs_min,
            "observability_trace": obs_trace,
            "observability_log_det": obs_log_det,
            "observability_condition_number": obs_cond,
            "observability_effective_rank": obs_eff_rank,
            "hankel_norm": hankel_norm,
            "hankel_min_singular_value": hankel_min_sv,
            "hankel_condition_number": hankel_cond,
            "hankel_effective_rank": hankel_eff_rank,
            "process_noise_trace": Q_trace,
            "process_noise_log_det": Q_log_det,
            "process_noise_condition_number": Q_cond,
            "measurement_noise_trace": R_trace,
            "measurement_noise_log_det": R_log_det,
            "measurement_noise_condition_number": R_cond,
            "steady_state_signal_to_noise_ratio": ss_snr,
            "filter_closed_loop_spectral_radius": filter_sr,
            "filter_covariance_trace": P_kf_trace,
            "filter_covariance_log_det": P_kf_log_det,
            "innovation_covariance_trace": innov_cov_trace,
            "log10_time_steps": math.log10(self.time_steps),
            "trials": float(self.trials),
            "log10_x0_norm": math.log10(x0_norm) if x0_norm > 0 else float("nan"),
            "log10_input_rms": math.log10(rms) if rms > 0 else float("nan"),
            "log10_excitation_to_process_noise_ratio": (
                math.log10((rms**2 * B_frob_sq) / Q_trace)
                if rms > 0 and Q_trace > 0
                else float("nan")
            ),
            "input_kind_id": float(self.input.KIND.id),
        }
