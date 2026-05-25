from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gg4_wk2.condition_library import ConditionLibrary

_CATEGORICAL_COL = "input_kind"
_VALID_SCORES = {"r2_x", "r2_u", "both"}


class EvaluationIllustrator:
    def __init__(
        self,
        library: ConditionLibrary,
        evaluation: pd.DataFrame,
        *,
        ncols: int = 6,
        figsize_per_cell: tuple[float, float] = (3.0, 2.5),
        alpha: float = 0.5,
        s: float = 10.0,
        color_r2_x: str = "steelblue",
        color_r2_u: str = "darkorange",
    ) -> None:
        if len(library.conditions) != len(evaluation):
            raise ValueError(
                f"library has {len(library.conditions)} conditions but evaluation has "
                f"{len(evaluation)} rows"
            )
        self.library = library
        self.evaluation = evaluation
        self.ncols = ncols
        self.figsize_per_cell = figsize_per_cell
        self.alpha = alpha
        self.s = s
        self._colors = {"r2_x": color_r2_x, "r2_u": color_r2_u}

    def plot(
        self,
        features: Sequence[str] | None = None,
        *,
        score: str = "both",
    ) -> matplotlib.figure.Figure:
        if score not in _VALID_SCORES:
            raise ValueError(f"score must be one of {_VALID_SCORES!r}, got {score!r}")

        all_cols = list(self.library.features.columns)
        if features is None:
            cols = all_cols
        else:
            unknown = [c for c in features if c not in all_cols]
            if unknown:
                raise ValueError(f"Unknown feature column(s): {unknown}")
            cols = list(features)

        score_names = ["r2_x", "r2_u"] if score == "both" else [score]
        scores: dict[str, pd.Series] = {  # type: ignore[type-arg]
            name: self.evaluation[name].reset_index(drop=True) for name in score_names
        }

        fig, axes_flat = self._make_grid(len(cols))
        for col, ax in zip(cols, axes_flat):
            x = self.library.features[col].reset_index(drop=True)
            if col == _CATEGORICAL_COL:
                self._plot_categorical_scatter(ax, x, col, scores)
            else:
                self._plot_numeric_scatter(ax, x, col, scores)

        fig.tight_layout()
        return fig

    def _make_grid(
        self, n: int
    ) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
        ncols = min(self.ncols, n)
        nrows = math.ceil(n / ncols)
        w, h = self.figsize_per_cell
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h))
        axes_flat: list[matplotlib.axes.Axes] = np.array(axes).flatten().tolist()
        for i in range(n, len(axes_flat)):
            axes_flat[i].set_visible(False)
        return fig, axes_flat[:n]

    def _plot_numeric_scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: pd.Series,  # type: ignore[type-arg]
        col: str,
        scores: dict[str, pd.Series],  # type: ignore[type-arg]
    ) -> None:
        x = x.replace([float("inf"), float("-inf")], float("nan"))
        valid_x = ~x.isna()
        r_parts: list[str] = []
        show_legend = len(scores) > 1

        for name, score_series in scores.items():
            mask = valid_x & ~score_series.isna()
            x_vals = x[mask].to_numpy(dtype=float)
            y_vals = score_series[mask].to_numpy(dtype=float)
            if len(x_vals) >= 2:
                r = float(np.corrcoef(x_vals, y_vals)[0, 1])
            else:
                r = float("nan")
            r_parts.append(f"{name} r={r:.2f}")
            ax.scatter(
                x_vals,
                y_vals,
                alpha=self.alpha,
                s=self.s,
                color=self._colors[name],
                label=name,
            )

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("R²", fontsize=7)
        ax.tick_params(labelsize=6)
        title = col + "\n" + " | ".join(r_parts)
        ax.set_title(title, fontsize=7)
        if show_legend:
            ax.legend(fontsize=6)

    def _plot_categorical_scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: pd.Series,  # type: ignore[type-arg]
        col: str,
        scores: dict[str, pd.Series],  # type: ignore[type-arg]
    ) -> None:
        categories = sorted(x.dropna().unique().tolist())
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        idx = x.map(cat_to_idx)

        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.15, 0.15, size=len(idx))
        x_jittered = idx.to_numpy(dtype=float) + jitter

        show_legend = len(scores) > 1
        for name, score_series in scores.items():
            mask = ~idx.isna() & ~score_series.isna()
            ax.scatter(
                x_jittered[mask.to_numpy()],
                score_series[mask].to_numpy(dtype=float),
                alpha=self.alpha,
                s=self.s,
                color=self._colors[name],
                label=name,
            )

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(list(range(len(categories))))
        ax.set_xticklabels(categories, rotation=30, fontsize=6)
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("R²", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{col}\n(categorical)", fontsize=7)
        if show_legend:
            ax.legend(fontsize=6)
