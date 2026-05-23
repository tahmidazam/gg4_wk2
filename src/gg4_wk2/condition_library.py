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
                    cond = Condition(
                        model=model,
                        x_0=x_0,
                        input=input,
                        time_steps=time_steps,
                        trials=trials,
                    )
                    row: dict[str, Any] = {
                        **cond.features,
                        "state_dim": cond.model.state_dim,
                        "input_dim": cond.model.input_dim,
                        "obs_dim": cond.model.obs_dim,
                        "input_kind": cond.input.KIND.value,
                    }
                except RuntimeError, ValueError, np.linalg.LinAlgError:
                    n_dropped += 1
                    continue

                conditions.append(cond)
                rows.append(row)
                pbar.update(1)

        return cls(
            conditions=tuple(conditions),
            features=pd.DataFrame(rows),
            n_dropped=n_dropped,
        )
