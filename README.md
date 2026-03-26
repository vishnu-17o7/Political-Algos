# Political-Algos

A collection of **population-based optimization algorithms** each inspired by a
different political ideology, plus a benchmark suite to compare them on classic
test functions.

---

## Algorithms

| Algorithm | Ideology | Core Mechanism |
|-----------|----------|----------------|
| `CommunismOptimizer` | Communism | All agents pulled equally towards the collective mean; no individual advantage |
| `CapitalismOptimizer` | Capitalism | Elites survive & reproduce with wealth-scaled mutation; weakest replaced by random entrants |
| `SocialismOptimizer` | Socialism | Progressive redistribution: elites share directional advantage with low performers; universal noise floor |
| `DemocracyOptimizer` | Democracy | Agents vote (weighted by rank) for a consensus search direction; periodic elections renew the population |
| `MonarchyOptimizer` | Monarchy | A single monarch guides all subjects; throne passes only when a challenger surpasses the ruler |

All algorithms share a common interface via `algorithms.base.BaseOptimizer`.

---

## Requirements

- Python ≥ 3.12
- `numpy`
- `matplotlib` *(optional – only needed for `--plot`)*
- `pytest` *(only needed to run tests)*

```bash
pip install numpy          # required
pip install matplotlib     # optional – for convergence plots
pip install pytest         # optional – for running tests
```

---

## Quick Start

```python
from algorithms import CapitalismOptimizer
import numpy as np

def sphere(x):
    return float(np.sum(x ** 2))

opt = CapitalismOptimizer(objective_func=sphere, dim=10, bounds=(-5, 5), seed=42)
best_solution, best_fitness = opt.optimize()
print(f"Best fitness: {best_fitness:.6f}")
```

---

## Benchmark

Run the benchmark comparing all five algorithms on five standard test functions
(Sphere, Rastrigin, Rosenbrock, Ackley, Griewank):

```bash
# Default settings (dim=10, 10 independent runs)
python benchmark.py

# Custom settings
python benchmark.py --dim 5 --runs 5 --iter 500

# Save convergence plots (requires matplotlib)
python benchmark.py --plot
```

### Example output

```
======================================================================
  Function: Sphere
======================================================================
Algorithm              Mean            Std           Best    Time(s)
----------------------------------------------------------------------
Communism          0.112447       0.031200       0.072739     0.0489
Capitalism         0.018961       0.011732       0.003042     0.1293
Socialism          0.000373       0.000073       0.000320     0.1214 ★
Democracy         10.314978       1.003443       8.936248     0.2951
Monarchy           0.006168       0.001599       0.003914     0.0906
...
```

★ marks the best mean fitness for each function.

---

## Tests

```bash
pytest tests/test_algorithms.py -v
```

59 tests covering the shared contract (return types, bounds, reproducibility,
monotone history, convergence) and algorithm-specific parameter checks.

---

## Project Structure

```
Political-Algos/
├── algorithms/
│   ├── __init__.py        # Package exports
│   ├── base.py            # Abstract BaseOptimizer
│   ├── communism.py       # CommunismOptimizer
│   ├── capitalism.py      # CapitalismOptimizer
│   ├── socialism.py       # SocialismOptimizer
│   ├── democracy.py       # DemocracyOptimizer
│   └── monarchy.py        # MonarchyOptimizer
├── tests/
│   └── test_algorithms.py # pytest test suite (59 tests)
├── benchmark.py           # Benchmark runner & display
└── README.md
```