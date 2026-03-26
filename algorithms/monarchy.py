"""Monarchy Optimizer.

Core ideology
-------------
*One sovereign rules absolutely; all subjects follow the crown.*

In this optimiser a single **monarch** (the all-time best solution found)
commands the population:

* All agents move toward the monarch with a loyalty-scaled step.
* The monarch is never directly modified – divine right of succession means
  the throne passes only when a subject demonstrably surpasses the current
  ruler.
* Subjects mutate around the monarch's position, and those that improve
  beyond the monarch take the throne (palace coup / succession).
* A small "rebel" fraction ignores the monarch entirely and explores
  randomly – representing dissidents who may accidentally discover a better
  solution.

This resembles a centre-guided search similar to CMA-ES or PSO with a
fixed global best, but with the monarchy narrative baked in.
"""

import numpy as np
from .base import BaseOptimizer


class MonarchyOptimizer(BaseOptimizer):
    """Population-based optimiser inspired by absolute monarchy.

    Parameters
    ----------
    loyalty:
        Fraction of each agent's step devoted to following the monarch (0–1).
    rebel_fraction:
        Fraction of population that ignores the monarch and explores freely.
    mutation_scale:
        Noise scale for subjects mutating around the monarch.
    """

    name = "Monarchy"

    def __init__(
        self,
        objective_func,
        dim: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
        population_size: int = 30,
        max_iter: int = 500,
        seed: int | None = None,
        loyalty: float = 0.6,
        rebel_fraction: float = 0.1,
        mutation_scale: float = 0.3,
    ) -> None:
        super().__init__(
            objective_func, dim, bounds, population_size, max_iter, seed
        )
        self.loyalty = loyalty
        self.rebel_fraction = rebel_fraction
        self.mutation_scale = mutation_scale
        self._monarch: np.ndarray | None = None
        self._monarch_fitness: float = float("inf")

    def _step(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.population_size
        n_rebels = max(1, int(n * self.rebel_fraction))
        lo, hi = self.bounds

        # Update monarch if a subject has surpassed the throne
        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < self._monarch_fitness:
            self._monarch = population[best_idx].copy()
            self._monarch_fitness = float(fitness[best_idx])

        monarch = self._monarch

        new_population = np.empty_like(population)

        # Loyal subjects: move toward the monarch + local noise
        for i in range(n - n_rebels):
            direction = monarch - population[i]
            noise = self.rng.standard_normal(self.dim) * self.mutation_scale
            new_population[i] = population[i] + self.loyalty * direction + noise

        # Rebels: explore freely (ignore the monarch)
        new_population[n - n_rebels:] = self.rng.uniform(lo, hi, (n_rebels, self.dim))

        new_population = self._clip(new_population)
        new_fitness = self._evaluate(new_population)
        return new_population, new_fitness
