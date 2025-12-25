from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """
    Immutable container for math results.
    Frozen makes it hashable and prevents accidental overwrites during analysis.
    """

    rho_history: np.ndarray
    rmse_history: np.ndarray
    jitter: float
    avg_rmse: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_rmse": self.avg_rmse,
            "total_jitter": self.jitter,
            "efficiency": 1.0 / (self.avg_rmse * self.jitter + 1e-10),
        }
