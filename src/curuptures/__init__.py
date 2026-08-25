"""
cuRuptures
GPU-accelerated change-point detection with CuPy.
"""

from .costs import CostL2
from .pelt import Pelt


__version__ = "0.1.0"


__all__ = [
    "CostL2",
    "Pelt",
]
