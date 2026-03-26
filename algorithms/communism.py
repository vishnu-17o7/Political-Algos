"""Communism Optimizer.

Core ideology
-------------
*From each according to their ability, to each according to their need.*

In this optimiser, all agents share their information equally: at each step
every agent is pulled towards the **collective mean** of the population.
Individual excellence is suppressed – no single agent is rewarded more than
another, and the step size is identical for everyone.  Resources (search
budget) are distributed uniformly.

This leads to fast initial convergence towards the centre of mass, but can
stagnate once the population collapses to the mean.  A small random
perturbation (representing grass-roots innovation despite central planning)
is added to prevent complete stagnation.
"""

import numpy as np
from .base import BaseOptimizer


class CommunismOptimizer(BaseOptimizer):
    """Population-based optimiser inspired by communist resource-sharing.

    Parameters
    ----------
    collective_weight:
        Strength of the pull towards the collective mean (0–1).
    perturbation_scale:
        Scale of the random noise added to prevent stagnation.
    """

    name = "Communism"

    def __init__(
        self,
        objective_func,
        dim: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
        population_size: int = 30,
        max_iter: int = 500,
        seed: int | None = None,
        collective_weight: float = 0.7,
        perturbation_scale: float = 0.1,
    ) -> None:
        super().__init__(
            objective_func, dim, bounds, population_size, max_iter, seed
        )
        self.collective_weight = collective_weight
        self.perturbation_scale = perturbation_scale

    def _step(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Collective mean – the "state plan"
        mean = population.mean(axis=0)

        # Every agent moves uniformly towards the collective mean
        noise = self.rng.standard_normal(population.shape) * self.perturbation_scale
        new_population = (
            population
            + self.collective_weight * (mean - population)
            + noise
        )
        new_population = self._clip(new_population)
        new_fitness = self._evaluate(new_population)
        return new_population, new_fitness
