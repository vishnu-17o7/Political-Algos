"""Capitalism Optimizer.

Core ideology
-------------
*Free competition: the strong prosper, the weak are replaced.*

Inspired by free-market capitalism:

* **Elitism** – the top-performing solutions (capital owners) survive
  unchanged across generations.
* **Wealth accumulation** – elite solutions are replicated and mutated to
  exploit promising regions (capital begets capital).
* **Creative destruction** – the worst-performing solutions are discarded
  entirely and replaced by new random entrants (market entry).
* **Inequality** – step-size scales with individual rank: better solutions
  receive a larger search radius (greater investment capital).

The result is an aggressive exploitation-first strategy that converges
quickly in smooth landscapes but may struggle with highly multimodal ones.
"""

import numpy as np
from .base import BaseOptimizer


class CapitalismOptimizer(BaseOptimizer):
    """Population-based optimiser inspired by free-market capitalism.

    Parameters
    ----------
    elite_fraction:
        Fraction of the population kept as elites (capital owners).
    mutation_scale:
        Base mutation magnitude applied to elites when reproducing.
    """

    name = "Capitalism"

    def __init__(
        self,
        objective_func,
        dim: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
        population_size: int = 30,
        max_iter: int = 500,
        seed: int | None = None,
        elite_fraction: float = 0.2,
        mutation_scale: float = 0.3,
    ) -> None:
        super().__init__(
            objective_func, dim, bounds, population_size, max_iter, seed
        )
        self.elite_fraction = elite_fraction
        self.mutation_scale = mutation_scale

    def _step(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.population_size
        n_elite = max(1, int(n * self.elite_fraction))

        # Sort by fitness (lower = better)
        order = np.argsort(fitness)
        elites = population[order[:n_elite]].copy()
        elite_fitness = fitness[order[:n_elite]].copy()

        new_population = np.empty_like(population)
        new_fitness = np.empty(n)

        # Keep elites unchanged (wealth preservation)
        new_population[:n_elite] = elites
        new_fitness[:n_elite] = elite_fitness

        # Fill the rest: elites reproduce with rank-scaled mutation
        for i in range(n_elite, n):
            # Pick a parent from the elite pool (richer elites reproduce more)
            parent_idx = self.rng.integers(0, n_elite)
            parent = elites[parent_idx]

            # Wealth-scaled mutation: top elite can invest more
            rank_factor = 1.0 + (n_elite - parent_idx) / n_elite
            offspring = parent + self.rng.standard_normal(self.dim) * self.mutation_scale * rank_factor
            offspring = self._clip(offspring.reshape(1, -1))[0]
            new_population[i] = offspring
            new_fitness[i] = self.objective_func(offspring)

        return new_population, new_fitness
