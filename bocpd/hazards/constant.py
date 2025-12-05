"""Constant hazard function implementation."""

import numpy as np

from bocpd.hazards.base import HazardFunction


class ConstantHazard(HazardFunction):
    """Constant hazard function.

    Assumes a constant probability of changepoint occurrence at each time step,
    regardless of the run length. This corresponds to an exponential distribution
    for the time between changepoints.

    If lambda_ is the hazard rate, the expected run length is 1/lambda_.
    """

    def __init__(self, lambda_: float) -> None:
        """Initialize the constant hazard function.

        Args:
            lambda_: The constant hazard rate (0 < lambda_ <= 1).
                    For example, lambda_=0.01 means an expected run length of 100.

        Raises:
            ValueError: If lambda_ is not in (0, 1].
        """
        if not (0 < lambda_ <= 1):
            raise ValueError(f"lambda_ must be in (0, 1], got {lambda_}")

        self.lambda_ = lambda_

    def compute(self, r: int) -> float:
        """Compute the constant hazard at run length r.

        Args:
            r: The run length (non-negative integer).

        Returns:
            The constant hazard probability lambda_.

        Raises:
            ValueError: If r is negative.
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return self.lambda_

    def compute_survival(self, r: int) -> float:
        """Compute the survival probability.

        Args:
            r: The run length (non-negative integer).

        Returns:
            The survival probability 1 - lambda_.
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return 1.0 - self.lambda_

    def compute_log(self, r: int) -> float:
        """Compute log of the hazard function.

        Args:
            r: The run length (non-negative integer).

        Returns:
            Log of hazard probability log(lambda_).
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return np.log(self.lambda_)

    def compute_log_survival(self, r: int) -> float:
        """Compute log of the survival function.

        Args:
            r: The run length (non-negative integer).

        Returns:
            Log of survival probability log(1 - lambda_).
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return np.log1p(-self.lambda_)  # log1p for numerical stability

    def __repr__(self) -> str:
        """String representation."""
        expected_run_length = 1.0 / self.lambda_
        return f"ConstantHazard(lambda_={self.lambda_:.4f}, expected_run_length={expected_run_length:.2f})"
