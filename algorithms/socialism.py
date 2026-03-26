"""Socialism Optimizer.

Core ideology
-------------
*To each according to their contribution – but the state redistributes a
share of the surplus to those who need it most.*

Socialism blends individual merit with collective support:

* **Progressive redistribution** – a portion of each elite solution's
  advantage is shared (broadcast) to below-average performers.
* **Merit-based survival** – solutions are still ranked and the worst
  performers are replaced, but less harshly than under capitalism.
* **Universal basic step** – every agent receives a minimum random
  perturbation so that even weak solutions have a chance to improve
  (social safety net).

This produces a balance between exploitation (driven by elites) and
exploration (universal noise + redistribution lifts low performers).
"""

import numpy as np
from .base import BaseOptimizer


class SocialismOptimizer(BaseOptimizer):
    """Population-based optimiser inspired by social-democratic redistribution.

    Parameters
    ----------
    elite_fraction:
        Fraction of top solutions considered "high earners".
    tax_rate:
        Fraction of the elite's directional advantage redistributed to
        below-median performers (0 = pure capitalism, 1 = pure communism).
    basic_noise:
        Universal minimum noise applied to every agent (social safety net).
    """

    name = "Socialism"

    def __init__(
        self,
        objective_func,
        dim: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
        population_size: int = 30,
        max_iter: int = 500,
        seed: int | None = None,
        elite_fraction: float = 0.3,
        tax_rate: float = 0.4,
        basic_noise: float = 0.05,
    ) -> None:
        super().__init__(
            objective_func, dim, bounds, population_size, max_iter, seed
        )
        self.elite_fraction = elite_fraction
        self.tax_rate = tax_rate
        self.basic_noise = basic_noise

    def _step(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.population_size
        n_elite = max(1, int(n * self.elite_fraction))

        order = np.argsort(fitness)
        sorted_pop = population[order]

        elite_mean = sorted_pop[:n_elite].mean(axis=0)   # "wealthy" centroid
        poor_mean = sorted_pop[n_elite:].mean(axis=0)    # "needy" centroid

        new_population = np.empty_like(population)

        for rank, idx in enumerate(order):
            agent = population[idx]
            if rank < n_elite:
                # Elite: move toward own region, share a portion toward the poor
                move = (elite_mean - agent) * (1.0 - self.tax_rate)
                redistributed = (poor_mean - agent) * self.tax_rate * 0.1
                new_population[rank] = agent + move + redistributed
            else:
                # Non-elite: benefit from redistribution towards elite mean
                move = (elite_mean - agent) * self.tax_rate * 0.5
                new_population[rank] = agent + move

            # Universal basic noise (safety net)
            new_population[rank] += (
                self.rng.standard_normal(self.dim) * self.basic_noise
            )

        new_population = self._clip(new_population)
        new_fitness = self._evaluate(new_population)
        return new_population, new_fitness
