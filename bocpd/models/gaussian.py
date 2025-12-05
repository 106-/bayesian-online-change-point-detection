"""Gaussian predictive model with Normal-Inverse-Gamma prior."""

from typing import Any

import numpy as np
from scipy import stats

from bocpd.models.base import PredictiveModel


class GaussianModel(PredictiveModel):
    """Gaussian likelihood with Normal-Inverse-Gamma conjugate prior.

    This model assumes observations come from a Gaussian distribution with unknown
    mean and variance. The prior is a Normal-Inverse-Gamma distribution, which
    maintains conjugacy and allows for analytical posterior updates.

    Hyperparameters:
        mu0: Prior mean of the Gaussian mean parameter.
        kappa0: Prior precision (inverse variance) scaling for the mean.
        alpha0: Prior shape parameter for the precision (inverse variance).
        beta0: Prior rate parameter for the precision.

    The predictive distribution is a Student's t-distribution with parameters
    derived from the hyperparameters.
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        """Initialize the Gaussian model with hyperparameters.

        Args:
            mu0: Prior mean (default: 0.0).
            kappa0: Prior precision scaling (default: 1.0, must be > 0).
            alpha0: Prior shape parameter (default: 1.0, must be > 0).
            beta0: Prior rate parameter (default: 1.0, must be > 0).

        Raises:
            ValueError: If kappa0, alpha0, or beta0 are not positive.
        """
        if kappa0 <= 0:
            raise ValueError(f"kappa0 must be positive, got {kappa0}")
        if alpha0 <= 0:
            raise ValueError(f"alpha0 must be positive, got {alpha0}")
        if beta0 <= 0:
            raise ValueError(f"beta0 must be positive, got {beta0}")

        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def fit_empirical(self, data: np.ndarray) -> None:
        """Initialize hyperparameters using empirical Bayes.

        Estimates hyperparameters from the data using sample statistics.
        Sets weakly informative priors centered around the empirical estimates.

        Args:
            data: Historical observations (1D array).

        Raises:
            ValueError: If data is empty or contains fewer than 2 observations.
        """
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        if len(data) < 2:
            raise ValueError("Need at least 2 observations for empirical fitting")

        # Compute sample statistics
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Set mu0 to sample mean
        self.mu0 = sample_mean

        # Set kappa0 to be weakly informative (equivalent to ~1 observation)
        self.kappa0 = 1.0

        # Set alpha0 and beta0 based on sample variance
        # Using method of moments: E[variance] = beta/(alpha-1) for alpha > 1
        # Set alpha0 = 2 (minimal value for finite mean) and beta0 accordingly
        self.alpha0 = 2.0
        self.beta0 = sample_var * (self.alpha0 - 1)

    def predict(self, x: float) -> float:
        """Compute log probability density under the predictive distribution.

        The predictive distribution is a Student's t-distribution with:
        - degrees of freedom: 2 * alpha0
        - location: mu0
        - scale: sqrt(beta0 * (kappa0 + 1) / (alpha0 * kappa0))

        Args:
            x: Observation to evaluate.

        Returns:
            Log probability density log p(x | hyperparameters).
        """
        # Student's t parameters
        df = 2 * self.alpha0
        loc = self.mu0
        scale = np.sqrt(self.beta0 * (self.kappa0 + 1) / (self.alpha0 * self.kappa0))

        # Compute log probability using scipy
        log_prob = stats.t.logpdf(x, df=df, loc=loc, scale=scale)

        return log_prob

    def update(self, x: float) -> "GaussianModel":
        """Update posterior given a new observation.

        Performs analytical Bayesian update using conjugacy.

        Args:
            x: New observation.

        Returns:
            New GaussianModel instance with updated hyperparameters.
        """
        # Posterior hyperparameters (conjugate update)
        kappa_new = self.kappa0 + 1
        mu_new = (self.kappa0 * self.mu0 + x) / kappa_new
        alpha_new = self.alpha0 + 0.5
        beta_new = (
            self.beta0
            + 0.5 * self.kappa0 * (x - self.mu0) ** 2 / kappa_new
        )

        return GaussianModel(
            mu0=mu_new,
            kappa0=kappa_new,
            alpha0=alpha_new,
            beta0=beta_new,
        )

    def copy(self) -> "GaussianModel":
        """Create a deep copy of the model.

        Returns:
            New GaussianModel instance with the same hyperparameters.
        """
        return GaussianModel(
            mu0=self.mu0,
            kappa0=self.kappa0,
            alpha0=self.alpha0,
            beta0=self.beta0,
        )

    def get_params(self) -> dict[str, Any]:
        """Get current hyperparameters.

        Returns:
            Dictionary with keys: mu0, kappa0, alpha0, beta0.
        """
        return {
            "mu0": self.mu0,
            "kappa0": self.kappa0,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
        }

    def set_params(self, **params: Any) -> None:
        """Set hyperparameters.

        Args:
            **params: Hyperparameters to set (mu0, kappa0, alpha0, beta0).

        Raises:
            ValueError: If parameters are invalid.
        """
        if "mu0" in params:
            self.mu0 = params["mu0"]
        if "kappa0" in params:
            if params["kappa0"] <= 0:
                raise ValueError("kappa0 must be positive")
            self.kappa0 = params["kappa0"]
        if "alpha0" in params:
            if params["alpha0"] <= 0:
                raise ValueError("alpha0 must be positive")
            self.alpha0 = params["alpha0"]
        if "beta0" in params:
            if params["beta0"] <= 0:
                raise ValueError("beta0 must be positive")
            self.beta0 = params["beta0"]

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"GaussianModel(mu0={self.mu0:.4f}, kappa0={self.kappa0:.4f}, "
            f"alpha0={self.alpha0:.4f}, beta0={self.beta0:.4f})"
        )
