"""Bayesian Online Changepoint Detection (BOCPD) library.

This library provides a practical implementation of Bayesian online changepoint
detection with support for various conjugate models and hazard functions.

Example:
    >>> import numpy as np
    >>> from bocpd import BOCPD, GaussianModel, ConstantHazard
    >>>
    >>> # Initialize model and detector
    >>> model = GaussianModel()
    >>> hazard = ConstantHazard(lambda_=0.01)  # Expected run length of 100
    >>> detector = BOCPD(model=model, hazard=hazard)
    >>>
    >>> # Fit on historical data
    >>> historical_data = np.random.randn(100)
    >>> detector.fit(historical_data)
    >>>
    >>> # Process new observations
    >>> for x in np.random.randn(50):
    ...     result = detector.update(x)
    ...     if result["changepoint_prob"] > 0.5:
    ...         print(f"Changepoint detected!")
"""

from bocpd.detector import BOCPD
from bocpd.hazards import ConstantHazard, HazardFunction
from bocpd.models import GaussianModel, PredictiveModel

__version__ = "0.1.0"

__all__ = [
    "BOCPD",
    "PredictiveModel",
    "GaussianModel",
    "HazardFunction",
    "ConstantHazard",
]
