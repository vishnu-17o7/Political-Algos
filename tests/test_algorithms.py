"""Tests for political optimization algorithms."""

import math
import numpy as np
import pytest

from algorithms import (
    CommunismOptimizer,
    CapitalismOptimizer,
    SocialismOptimizer,
    DemocracyOptimizer,
    MonarchyOptimizer,
)
from algorithms.base import BaseOptimizer

# ---------------------------------------------------------------------------
# Simple test functions
# ---------------------------------------------------------------------------

def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)))


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

OPTIMIZERS = [
    CommunismOptimizer,
    CapitalismOptimizer,
    SocialismOptimizer,
    DemocracyOptimizer,
    MonarchyOptimizer,
]

COMMON_KWARGS = dict(
    dim=2,
    bounds=(-5.0, 5.0),
    population_size=20,
    max_iter=200,
    seed=0,
)


# ---------------------------------------------------------------------------
# Base-class contract tests (parameterised over all optimisers)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS, ids=lambda c: c.name)
class TestOptimizerContract:
    """Every optimiser must satisfy the base-class contract."""

    def test_is_subclass_of_base(self, OptimizerClass):
        assert issubclass(OptimizerClass, BaseOptimizer)

    def test_has_name_attribute(self, OptimizerClass):
        assert isinstance(OptimizerClass.name, str)
        assert len(OptimizerClass.name) > 0

    def test_optimize_returns_array_and_float(self, OptimizerClass):
        opt = OptimizerClass(objective_func=sphere, **COMMON_KWARGS)
        solution, fitness = opt.optimize()
        assert isinstance(solution, np.ndarray)
        assert solution.shape == (COMMON_KWARGS["dim"],)
        assert isinstance(fitness, float)

    def test_history_length(self, OptimizerClass):
        opt = OptimizerClass(objective_func=sphere, **COMMON_KWARGS)
        opt.optimize()
        assert len(opt.history) == COMMON_KWARGS["max_iter"]

    def test_history_is_non_increasing(self, OptimizerClass):
        """Best fitness should never get worse over time."""
        opt = OptimizerClass(objective_func=sphere, **COMMON_KWARGS)
        opt.optimize()
        for prev, curr in zip(opt.history, opt.history[1:]):
            assert curr <= prev + 1e-12, (
                f"{OptimizerClass.name}: history increased from {prev} to {curr}"
            )

    def test_solution_within_bounds(self, OptimizerClass):
        lo, hi = COMMON_KWARGS["bounds"]
        opt = OptimizerClass(objective_func=sphere, **COMMON_KWARGS)
        solution, _ = opt.optimize()
        assert np.all(solution >= lo - 1e-12)
        assert np.all(solution <= hi + 1e-12)

    def test_fitness_equals_objective_of_solution(self, OptimizerClass):
        opt = OptimizerClass(objective_func=sphere, **COMMON_KWARGS)
        solution, fitness = opt.optimize()
        assert abs(fitness - sphere(solution)) < 1e-12

    def test_reproducibility_with_seed(self, OptimizerClass):
        kwargs = dict(objective_func=sphere, **COMMON_KWARGS)
        opt1 = OptimizerClass(**kwargs)
        opt2 = OptimizerClass(**kwargs)
        _, fit1 = opt1.optimize()
        _, fit2 = opt2.optimize()
        assert fit1 == fit2

    def test_different_seeds_give_same_or_different_results(self, OptimizerClass):
        """Two different seeds should not raise; results are allowed to differ."""
        opt1 = OptimizerClass(objective_func=sphere, seed=1, **{k: v for k, v in COMMON_KWARGS.items() if k != "seed"})
        opt2 = OptimizerClass(objective_func=sphere, seed=99, **{k: v for k, v in COMMON_KWARGS.items() if k != "seed"})
        sol1, _ = opt1.optimize()
        sol2, _ = opt2.optimize()
        # Just check shapes are correct – values may legitimately differ
        assert sol1.shape == sol2.shape

    def test_convergence_on_sphere(self, OptimizerClass):
        """All algorithms should find a reasonably good solution on Sphere(2D)."""
        opt = OptimizerClass(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=30,
            max_iter=500,
            seed=42,
        )
        _, fitness = opt.optimize()
        assert fitness < 1.0, (
            f"{OptimizerClass.name} failed to find f < 1.0 on Sphere(2D), got {fitness:.4f}"
        )


# ---------------------------------------------------------------------------
# Individual algorithm-specific tests
# ---------------------------------------------------------------------------

class TestCommunismOptimizer:
    def test_custom_params(self):
        opt = CommunismOptimizer(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=20,
            max_iter=10,
            seed=0,
            collective_weight=0.5,
            perturbation_scale=0.2,
        )
        assert opt.collective_weight == 0.5
        assert opt.perturbation_scale == 0.2
        opt.optimize()  # should not raise


class TestCapitalismOptimizer:
    def test_custom_params(self):
        opt = CapitalismOptimizer(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=20,
            max_iter=10,
            seed=0,
            elite_fraction=0.3,
            mutation_scale=0.5,
        )
        assert opt.elite_fraction == 0.3
        assert opt.mutation_scale == 0.5
        opt.optimize()


class TestSocialismOptimizer:
    def test_custom_params(self):
        opt = SocialismOptimizer(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=20,
            max_iter=10,
            seed=0,
            elite_fraction=0.4,
            tax_rate=0.5,
            basic_noise=0.1,
        )
        assert opt.tax_rate == 0.5
        opt.optimize()


class TestDemocracyOptimizer:
    def test_custom_params(self):
        opt = DemocracyOptimizer(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=20,
            max_iter=10,
            seed=0,
            neighbourhood_size=3,
            step_size=0.3,
        )
        assert opt.step_size == 0.3
        opt.optimize()

    def test_neighbourhood_clamped_to_population(self):
        """neighbourhood_size must not exceed population_size - 1."""
        opt = DemocracyOptimizer(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=5,
            max_iter=5,
            seed=0,
            neighbourhood_size=100,  # larger than pop
        )
        assert opt.neighbourhood_size <= opt.population_size - 1


class TestMonarchyOptimizer:
    def test_custom_params(self):
        opt = MonarchyOptimizer(
            objective_func=sphere,
            dim=2,
            bounds=(-5.0, 5.0),
            population_size=20,
            max_iter=10,
            seed=0,
            loyalty=0.8,
            rebel_fraction=0.2,
            mutation_scale=0.2,
        )
        assert opt.loyalty == 0.8
        opt.optimize()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

class TestBenchmarkHelpers:
    """Smoke-tests for the benchmark module."""

    def test_benchmark_runs_without_error(self):
        from benchmark import run_benchmark
        results = run_benchmark(dim=2, population_size=10, max_iter=20, runs=2, seed=0)
        assert isinstance(results, dict)
        for func_name, algo_results in results.items():
            for algo_name, stats in algo_results.items():
                assert "mean" in stats
                assert "best" in stats
                assert math.isfinite(stats["mean"])

    def test_all_functions_present(self):
        from benchmark import TEST_FUNCTIONS
        expected = {"Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"}
        assert set(TEST_FUNCTIONS.keys()) == expected

    def test_all_optimizers_present(self):
        from benchmark import OPTIMIZERS
        names = {cls.name for cls in OPTIMIZERS}
        expected = {"Communism", "Capitalism", "Socialism", "Democracy", "Monarchy"}
        assert names == expected
