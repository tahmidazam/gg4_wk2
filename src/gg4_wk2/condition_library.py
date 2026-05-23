from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from gg4_wk2.condition import Condition
from gg4_wk2.sampler import (
    IntegerSampler,
    LinearGaussianModelSampler,
    WaveformSampler,
)


@dataclass(frozen=True)
class ConditionLibrary:
    conditions: tuple[Condition, ...] = field(repr=False)
    features: pd.DataFrame = field(repr=False)
    n_dropped: int

    @classmethod
    def from_samplers(
        cls,
        *,
        model_sampler: LinearGaussianModelSampler = LinearGaussianModelSampler(),
        time_steps_sampler: IntegerSampler = IntegerSampler(
            low=100, high=2000, log=True
        ),
        trials_sampler: IntegerSampler = IntegerSampler(low=1, high=10, log=True),
        x0_sampler: WaveformSampler = WaveformSampler(),
        input_sampler: WaveformSampler = WaveformSampler(),
        n_samples: int = 100,
        seed: int | None = None,
    ) -> ConditionLibrary:
        rng = np.random.default_rng(seed)

        conditions: list[Condition] = []
        rows: list[dict[str, Any]] = []
        n_dropped = 0

        with tqdm(total=n_samples, desc="Generating conditions") as pbar:
            while len(conditions) < n_samples:
                try:
                    model = model_sampler.sample(rng)
                    time_steps = time_steps_sampler.sample(rng)
                    trials = trials_sampler.sample(rng)
                    x_0 = x0_sampler.sample(
                        rng, time_steps=1, input_dim=model.state_dim
                    )
                    input = input_sampler.sample(
                        rng, time_steps=time_steps, input_dim=model.input_dim
                    )
                    condition = Condition(
                        model=model,
                        x_0=x_0,
                        input=input,
                        time_steps=time_steps,
                        trials=trials,
                    )
                    row: dict[str, Any] = {
                        **condition.features,
                        "state_dim": condition.model.state_dim,
                        "input_dim": condition.model.input_dim,
                        "obs_dim": condition.model.obs_dim,
                        "input_kind": condition.input.KIND.value,
                    }
                except RuntimeError, ValueError, np.linalg.LinAlgError:
                    n_dropped += 1
                    continue

                conditions.append(condition)
                rows.append(row)
                pbar.update(1)

        return cls(
            conditions=tuple(conditions),
            features=pd.DataFrame(rows),
            n_dropped=n_dropped,
        )
