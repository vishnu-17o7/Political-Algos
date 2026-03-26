"""Base class for all political optimization algorithms."""

import abc
import numpy as np


class BaseOptimizer(abc.ABC):
    """Abstract base class that every political optimizer must implement.

    Parameters
    ----------
    objective_func:
        A callable ``f(x) -> float`` to minimise.  Lower is better.
    dim:
        Dimensionality of the search space.
    bounds:
        A pair ``(lower, upper)`` applied uniformly to all dimensions.
    population_size:
        Number of candidate solutions maintained in the population.
    max_iter:
        Maximum number of iterations (generations) to run.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        objective_func,
        dim: int,
        bounds: tuple[float, float] = (-5.0, 5.0),
        population_size: int = 30,
        max_iter: int = 500,
        seed: int | None = None,
    ) -> None:
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.best_solution: np.ndarray | None = None
        self.best_fitness: float = float("inf")
        self.history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> tuple[np.ndarray, float]:
        """Run the optimization loop.

        Returns
        -------
        best_solution:
            The best candidate solution found.
        best_fitness:
            The objective value of ``best_solution``.
        """
        population = self._init_population()
        fitness = self._evaluate(population)
        self._update_best(population, fitness)

        for _ in range(self.max_iter):
            population, fitness = self._step(population, fitness)
            self._update_best(population, fitness)
            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness

    # ------------------------------------------------------------------
    # Helpers used by subclasses
    # ------------------------------------------------------------------

    def _init_population(self) -> np.ndarray:
        lo, hi = self.bounds
        return self.rng.uniform(lo, hi, (self.population_size, self.dim))

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.objective_func(ind) for ind in population])

    def _clip(self, population: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds
        return np.clip(population, lo, hi)

    def _update_best(self, population: np.ndarray, fitness: np.ndarray) -> None:
        idx = int(np.argmin(fitness))
        if fitness[idx] < self.best_fitness:
            self.best_fitness = float(fitness[idx])
            self.best_solution = population[idx].copy()

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _step(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one generation / iteration and return updated population and fitness."""
