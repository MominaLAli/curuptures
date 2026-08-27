"""
cuRuptures
GPU-accelerated change-point detection with CuPy.
"""

from .costs import CostL2, BatchCostL2
from .pelt import Pelt
from .batch_pelt import BatchPelt
from .batch_optimal_partitioning import BatchOptimalPartitioning


__version__ = "0.1.1"


__all__ = [
    "CostL2",
    "BatchCostL2",
    "Pelt",
    "BatchPelt",
    "BatchOptimalPartitioning",
]
