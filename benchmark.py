"""Benchmark suite that compares all political optimization algorithms.

Test functions (minimisation, global minimum = 0):

* **Sphere**      – f(x) = Σ xᵢ²                       (unimodal, easy)
* **Rastrigin**   – f(x) = 10n + Σ [xᵢ² - 10cos(2πxᵢ)] (highly multimodal)
* **Rosenbrock**  – f(x) = Σ [100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²] (narrow valley)
* **Ackley**      – f(x) = -20exp(…) - exp(…) + 20 + e  (multimodal)
* **Griewank**    – f(x) = Σ xᵢ²/4000 - Π cos(xᵢ/√i) + 1 (many local minima)

Usage
-----
    python benchmark.py                  # full benchmark, table output
    python benchmark.py --dim 5          # override dimensionality
    python benchmark.py --runs 5         # independent runs per algorithm
    python benchmark.py --plot           # save convergence plots
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable

import numpy as np

from algorithms import (
    CommunismOptimizer,
    CapitalismOptimizer,
    SocialismOptimizer,
    DemocracyOptimizer,
    MonarchyOptimizer,
)

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    a, b, c = 20.0, 0.2, 2 * math.pi
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    return float(
        -a * math.exp(-b * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + a
        + math.e
    )


def griewank(x: np.ndarray) -> float:
    sum_sq = np.sum(x ** 2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_sq - prod_cos + 1.0)


TEST_FUNCTIONS: dict[str, tuple[Callable, tuple[float, float]]] = {
    "Sphere":     (sphere,     (-5.12, 5.12)),
    "Rastrigin":  (rastrigin,  (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-2.048, 2.048)),
    "Ackley":     (ackley,     (-32.768, 32.768)),
    "Griewank":   (griewank,   (-600.0, 600.0)),
}

OPTIMIZERS = [
    CommunismOptimizer,
    CapitalismOptimizer,
    SocialismOptimizer,
    DemocracyOptimizer,
    MonarchyOptimizer,
]

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    dim: int = 10,
    population_size: int = 40,
    max_iter: int = 500,
    runs: int = 10,
    seed: int = 42,
) -> dict:
    """Run all algorithms on all test functions and return results dict."""
    results: dict[str, dict[str, dict]] = {}

    for func_name, (func, bounds) in TEST_FUNCTIONS.items():
        results[func_name] = {}
        for OptimizerClass in OPTIMIZERS:
            algo_name = OptimizerClass.name
            best_values: list[float] = []
            elapsed_times: list[float] = []
            histories: list[list[float]] = []

            for run in range(runs):
                opt = OptimizerClass(
                    objective_func=func,
                    dim=dim,
                    bounds=bounds,
                    population_size=population_size,
                    max_iter=max_iter,
                    seed=seed + run,
                )
                t0 = time.perf_counter()
                _, best_val = opt.optimize()
                elapsed = time.perf_counter() - t0

                best_values.append(best_val)
                elapsed_times.append(elapsed)
                histories.append(opt.history)

            results[func_name][algo_name] = {
                "mean": float(np.mean(best_values)),
                "std": float(np.std(best_values)),
                "best": float(np.min(best_values)),
                "worst": float(np.max(best_values)),
                "mean_time": float(np.mean(elapsed_times)),
                "histories": histories,
            }

    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _col_width(col: str, data: list[str]) -> int:
    return max(len(col), *(len(d) for d in data))


def print_results(results: dict) -> None:
    algo_names = [cls.name for cls in OPTIMIZERS]

    for func_name, algo_results in results.items():
        print(f"\n{'='*70}")
        print(f"  Function: {func_name}")
        print(f"{'='*70}")

        header = f"{'Algorithm':<16} {'Mean':>14} {'Std':>14} {'Best':>14} {'Time(s)':>10}"
        print(header)
        print("-" * len(header))

        # Determine the winner (lowest mean)
        winner = min(algo_names, key=lambda a: algo_results[a]["mean"])

        for algo in algo_names:
            r = algo_results[algo]
            marker = " ★" if algo == winner else ""
            print(
                f"{algo:<16} {r['mean']:>14.6f} {r['std']:>14.6f} "
                f"{r['best']:>14.6f} {r['mean_time']:>10.4f}{marker}"
            )


def save_plots(results: dict, dim: int) -> None:
    """Save convergence plots – only attempted if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[info] matplotlib not found – skipping plots.")
        return

    algo_names = [cls.name for cls in OPTIMIZERS]
    n_funcs = len(TEST_FUNCTIONS)
    fig, axes = plt.subplots(1, n_funcs, figsize=(5 * n_funcs, 4))
    if n_funcs == 1:
        axes = [axes]

    for ax, (func_name, algo_results) in zip(axes, results.items()):
        for algo in algo_names:
            histories = algo_results[algo]["histories"]
            # Average convergence curve across runs
            avg_history = np.mean(histories, axis=0)
            ax.semilogy(avg_history, label=algo)
        ax.set_title(func_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness (log scale)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Political Optimizers – Convergence (dim={dim})", fontsize=13)
    plt.tight_layout()
    fname = "convergence_plots.png"
    plt.savefig(fname, dpi=120)
    print(f"\n[info] Convergence plots saved to '{fname}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark political optimization algorithms."
    )
    parser.add_argument("--dim", type=int, default=10, help="Search space dimensionality")
    parser.add_argument("--pop", type=int, default=40, help="Population size")
    parser.add_argument("--iter", type=int, default=500, help="Max iterations")
    parser.add_argument("--runs", type=int, default=10, help="Independent runs per algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--plot", action="store_true", help="Save convergence plots")
    args = parser.parse_args()

    print(
        f"\nRunning benchmark  |  dim={args.dim}  pop={args.pop}  "
        f"iter={args.iter}  runs={args.runs}"
    )
    print("Please wait…\n")

    results = run_benchmark(
        dim=args.dim,
        population_size=args.pop,
        max_iter=args.iter,
        runs=args.runs,
        seed=args.seed,
    )

    print_results(results)

    if args.plot:
        save_plots(results, args.dim)

    print(f"\n{'='*70}")
    print("  Legend: ★ = best mean fitness for this function")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
