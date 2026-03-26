"""Political Optimization Algorithms package."""

from .communism import CommunismOptimizer
from .capitalism import CapitalismOptimizer
from .socialism import SocialismOptimizer
from .democracy import DemocracyOptimizer
from .monarchy import MonarchyOptimizer

__all__ = [
    "CommunismOptimizer",
    "CapitalismOptimizer",
    "SocialismOptimizer",
    "DemocracyOptimizer",
    "MonarchyOptimizer",
]
