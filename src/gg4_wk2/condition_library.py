from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gg4_wk2.condition import Condition
from gg4_wk2.sampler import (
    IntegerSampler,
    LinearGaussianModelSampler,
    WaveformSampler,
)


@dataclass(frozen=True)
class ConditionLibrary:
    conditions: tuple[Condition, ...] = field(repr=False)
    properties: NDArray[np.float64] = field(repr=False)
    dims: NDArray[np.int64] = field(repr=False)
    input_kinds: tuple[str, ...] = field(repr=False)
    property_names: tuple[str, ...]
    n_dropped: int

    @classmethod
    def from_samplers(
        cls,
        *,
        model_sampler: LinearGaussianModelSampler,
        time_steps_sampler: IntegerSampler,
        trials_sampler: IntegerSampler,
        x0_sampler: WaveformSampler,
        input_sampler: WaveformSampler,
        n_samples: int,
        seed: int | None = None,
    ) -> "ConditionLibrary":
        rng = np.random.default_rng(seed)

        conditions: list[Condition] = []
        property_rows: list[NDArray[np.float64]] = []
        dim_rows: list[tuple[int, int, int]] = []
        kinds: list[str] = []
        n_dropped = 0

        while len(conditions) < n_samples:
            try:
                model = model_sampler.sample(rng)
                time_steps = time_steps_sampler.sample(rng)
                trials = trials_sampler.sample(rng)
                x_0 = x0_sampler.sample(rng, time_steps=1, input_dim=model.state_dim)
                input = input_sampler.sample(
                    rng, time_steps=time_steps, input_dim=model.input_dim
                )
                cond = Condition(
                    model=model,
                    x_0=x_0,
                    input=input,
                    time_steps=time_steps,
                    trials=trials,
                )
                vec = cond.scalar_property_vector
            except RuntimeError, ValueError, np.linalg.LinAlgError:
                n_dropped += 1
                continue

            conditions.append(cond)
            property_rows.append(vec)
            dim_rows.append(
                (cond.model.state_dim, cond.model.input_dim, cond.model.obs_dim)
            )
            kinds.append(cond.input.KIND.value)

        properties = np.vstack(property_rows).astype(np.float64)
        dims = np.asarray(dim_rows, dtype=np.int64)
        properties.setflags(write=False)
        dims.setflags(write=False)

        return cls(
            conditions=tuple(conditions),
            properties=properties,
            dims=dims,
            input_kinds=tuple(kinds),
            property_names=Condition.SCALAR_PROPERTY_NAMES,
            n_dropped=n_dropped,
        )

    @property
    def n_conditions(self) -> int:
        return self.properties.shape[0]

    def column(self, name: str) -> NDArray[np.float64]:
        idx = self.property_names.index(name)
        return self.properties[:, idx]

    def filter_by_kind(self, kind: str) -> "ConditionLibrary":
        mask = np.array([k == kind for k in self.input_kinds], dtype=bool)
        return ConditionLibrary(
            conditions=tuple(c for c, keep in zip(self.conditions, mask) if keep),
            properties=self.properties[mask],
            dims=self.dims[mask],
            input_kinds=tuple(k for k, keep in zip(self.input_kinds, mask) if keep),
            property_names=self.property_names,
            n_dropped=0,
        )

    def __repr__(self) -> str:
        return (
            f"ConditionLibrary(n_conditions={self.n_conditions}, "
            f"n_properties={len(self.property_names)}, "
            f"n_dropped={self.n_dropped})"
        )
