"""Democracy Optimizer.

Core ideology
-------------
*The will of the majority determines the collective direction.*

In this optimiser, every solution casts a "vote" for a search direction:

* Each agent identifies its **best local neighbour** (the agent in its
  neighbourhood that has the lowest fitness) and votes for the direction
  towards that neighbour.
* Votes are **weighted by fitness rank** – better solutions have louder
  voices (representative democracy with a meritocratic twist).
* The weighted majority vector is broadcast as the consensus direction.
* A small independent noise term represents individual freedom of
  expression (civil liberties).
* Periodically, a small fraction of agents is re-initialised randomly
  to represent new political entrants / elections refreshing representation.
"""

import numpy as np
from .base import BaseOptimizer


class DemocracyOptimizer(BaseOptimizer):
    """Population-based optimiser inspired by democratic consensus voting.

    Parameters
    ----------
    neighbourhood_size:
        Number of peers each agent compares itself against when voting.
    step_size:
        Magnitude of movement along the consensus direction.
    noise_scale:
        Individual freedom noise added after consensus movement.
    election_interval:
        Every this many iterations, re-initialise a fraction of the
        population to simulate periodic elections / renewal.
    renewal_fraction:
        Fraction of population replaced during an election cycle.
    """

    name = "Democracy"

    def __init__(
        self,
        objective_func,
        dim: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
        population_size: int = 30,
        max_iter: int = 500,
        seed: int | None = None,
        neighbourhood_size: int = 5,
        step_size: float = 0.4,
        noise_scale: float = 0.05,
        election_interval: int = 50,
        renewal_fraction: float = 0.1,
    ) -> None:
        super().__init__(
            objective_func, dim, bounds, population_size, max_iter, seed
        )
        self.neighbourhood_size = min(neighbourhood_size, population_size - 1)
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.election_interval = election_interval
        self.renewal_fraction = renewal_fraction
        self._iter_count = 0

    def _step(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._iter_count += 1
        n = self.population_size
        order = np.argsort(fitness)

        # Fitness-rank weights (rank 0 = best gets highest weight)
        weights = np.empty(n)
        for rank, idx in enumerate(order):
            weights[idx] = n - rank  # best agent gets weight n, worst gets 1

        # Each agent votes for the direction towards its best neighbour
        votes = np.zeros((n, self.dim))
        for i in range(n):
            # Sample neighbourhood (excluding self)
            candidates = list(range(n))
            candidates.remove(i)
            neighbours = self.rng.choice(candidates, self.neighbourhood_size, replace=False)
            best_nb = neighbours[np.argmin(fitness[neighbours])]
            direction = population[best_nb] - population[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                votes[i] = direction / norm

        # Weighted consensus direction
        consensus = (weights[:, None] * votes).sum(axis=0)
        c_norm = np.linalg.norm(consensus)
        if c_norm > 1e-10:
            consensus /= c_norm

        # Every agent moves along consensus + individual noise
        noise = self.rng.standard_normal(population.shape) * self.noise_scale
        new_population = population + self.step_size * consensus + noise
        new_population = self._clip(new_population)
        new_fitness = self._evaluate(new_population)

        # Periodic elections: replace worst agents with fresh candidates
        if self._iter_count % self.election_interval == 0:
            n_renew = max(1, int(n * self.renewal_fraction))
            worst_indices = np.argsort(new_fitness)[-n_renew:]
            lo, hi = self.bounds
            new_population[worst_indices] = self.rng.uniform(lo, hi, (n_renew, self.dim))
            new_fitness[worst_indices] = self._evaluate(new_population[worst_indices])

        return new_population, new_fitness
