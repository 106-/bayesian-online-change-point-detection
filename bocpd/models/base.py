"""Base class for predictive models in Bayesian online changepoint detection."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PredictiveModel(ABC):
    """Abstract base class for Bayesian conjugate predictive models.

    All predictive models must implement this interface to be used with BOCPD.
    Models should maintain hyperparameters for a conjugate prior distribution
    and provide methods for empirical Bayes initialization, prediction, and updating.
    """

    @abstractmethod
    def fit_empirical(self, data: np.ndarray) -> None:
        """Initialize hyperparameters using empirical Bayes from historical data.

        This method estimates reasonable hyperparameters from the observed data,
        typically by computing sample statistics and setting weakly informative priors.

        Args:
            data: Historical observations as a 1D numpy array.

        Raises:
            ValueError: If data is empty or invalid.
        """
        pass

    @abstractmethod
    def predict(self, x: float) -> float:
        """Compute log probability density of observation x under the predictive distribution.

        The predictive distribution is the posterior predictive distribution given
        the current hyperparameters (i.e., integrating out the latent parameters).

        Args:
            x: The observation to evaluate.

        Returns:
            Log probability density log p(x | D_t) where D_t is the data seen so far.
        """
        pass

    @abstractmethod
    def update(self, x: float) -> "PredictiveModel":
        """Update the posterior distribution given a new observation.

        This method performs a Bayesian update using the conjugacy property,
        analytically computing the posterior hyperparameters.

        IMPORTANT: This method should return a NEW instance with updated parameters,
        leaving the original instance unchanged (immutability).

        Args:
            x: The new observation.

        Returns:
            A new PredictiveModel instance with updated hyperparameters.
        """
        pass

    @abstractmethod
    def copy(self) -> "PredictiveModel":
        """Create a deep copy of the model.

        Returns:
            A new PredictiveModel instance with the same hyperparameters.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get current hyperparameters as a dictionary.

        Returns:
            Dictionary containing all hyperparameters.
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> None:
        """Set hyperparameters from a dictionary.

        Args:
            **params: Hyperparameters to set.

        Raises:
            ValueError: If parameters are invalid.
        """
        pass
