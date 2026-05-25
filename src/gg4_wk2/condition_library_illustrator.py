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
_INT_COLS = {"state_dim", "input_dim", "obs_dim"}


class ConditionLibraryIllustrator:
    def __init__(
        self,
        library: ConditionLibrary,
        *,
        ncols: int = 6,
        figsize_per_cell: tuple[float, float] = (3.0, 2.5),
        bins: int = 30,
        color: str = "steelblue",
        edgecolor: str = "white",
    ) -> None:
        self.library = library
        self.ncols = ncols
        self.figsize_per_cell = figsize_per_cell
        self.bins = bins
        self.color = color
        self.edgecolor = edgecolor

    def plot(self, features: Sequence[str] | None = None) -> matplotlib.figure.Figure:
        all_cols = list(self.library.features.columns)
        if features is None:
            cols = all_cols
        else:
            unknown = [c for c in features if c not in all_cols]
            if unknown:
                raise ValueError(f"Unknown feature column(s): {unknown}")
            cols = list(features)

        fig, axes_flat = self._make_grid(len(cols))
        for col, ax in zip(cols, axes_flat):
            series = self.library.features[col]
            if col == _CATEGORICAL_COL:
                self._plot_categorical(ax, series, col)
            else:
                self._plot_numeric(ax, series, col)

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

    def _plot_numeric(
        self,
        ax: matplotlib.axes.Axes,
        series: pd.Series,
        col: str,  # type: ignore[type-arg]
    ) -> None:
        valid = (
            series.dropna()
            .replace([float("inf"), float("-inf")], float("nan"))
            .dropna()
        )
        ax.set_title(col, fontsize=8)
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=6)
        if valid.empty:
            ax.text(
                0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center"
            )
            return
        if col in _INT_COLS:
            n_unique = int(valid.nunique())
            b = min(n_unique, self.bins)
        else:
            b = self.bins
        ax.hist(valid.to_numpy(), bins=b, color=self.color, edgecolor=self.edgecolor)

    def _plot_categorical(
        self,
        ax: matplotlib.axes.Axes,
        series: pd.Series,
        col: str,  # type: ignore[type-arg]
    ) -> None:
        counts = series.value_counts().sort_index()
        ax.bar(
            counts.index.astype(str),
            counts.values,
            color=self.color,
            edgecolor=self.edgecolor,
        )
        ax.set_title(col, fontsize=8)
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=6)
        if len(counts) > 3:
            ax.tick_params(axis="x", rotation=30)
