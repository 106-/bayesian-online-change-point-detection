"""Base class for hazard functions in Bayesian online changepoint detection."""

from abc import ABC, abstractmethod


class HazardFunction(ABC):
    """Abstract base class for hazard functions.

    A hazard function specifies the probability of a changepoint occurring
    at each run length (time since the last changepoint). Different hazard
    functions encode different assumptions about the distribution of run lengths.
    """

    @abstractmethod
    def compute(self, r: int) -> float:
        """Compute the hazard function at run length r.

        The hazard function h(r) represents the probability that a changepoint
        occurs at time t given that the current run length is r.

        Args:
            r: The run length (non-negative integer).

        Returns:
            The hazard probability h(r) in [0, 1].

        Raises:
            ValueError: If r is negative.
        """
        pass

    def compute_survival(self, r: int) -> float:
        """Compute the survival function at run length r.

        The survival function S(r) = 1 - h(r) represents the probability
        that no changepoint occurs. This method can be overridden for efficiency.

        Args:
            r: The run length (non-negative integer).

        Returns:
            The survival probability S(r) in [0, 1].
        """
        return 1.0 - self.compute(r)

    def compute_log(self, r: int) -> float:
        """Compute log of the hazard function at run length r.

        This method can be overridden for numerical stability.

        Args:
            r: The run length (non-negative integer).

        Returns:
            Log of hazard probability log(h(r)).
        """
        import numpy as np

        hazard = self.compute(r)
        if hazard == 0:
            return -np.inf
        return np.log(hazard)

    def compute_log_survival(self, r: int) -> float:
        """Compute log of the survival function at run length r.

        This method can be overridden for numerical stability.

        Args:
            r: The run length (non-negative integer).

        Returns:
            Log of survival probability log(S(r)).
        """
        import numpy as np

        survival = self.compute_survival(r)
        if survival == 0:
            return -np.inf
        return np.log(survival)
