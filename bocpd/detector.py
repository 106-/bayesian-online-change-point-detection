"""Bayesian Online Changepoint Detection (BOCPD) implementation."""

from typing import Optional

import numpy as np

from bocpd.hazards.base import HazardFunction
from bocpd.models.base import PredictiveModel


class BOCPD:
    """Bayesian Online Changepoint Detection.

    This class implements the BOCPD algorithm for detecting changepoints in
    sequential data. It maintains a distribution over run lengths (time since
    the last changepoint) and updates it online as new observations arrive.

    Reference:
        Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
        arXiv preprint arXiv:0710.3742.
    """

    def __init__(
        self,
        model: PredictiveModel,
        hazard: HazardFunction,
        max_run_length: Optional[int] = None,
    ) -> None:
        """Initialize the BOCPD detector.

        Args:
            model: Predictive model (will be copied for each run length).
            hazard: Hazard function specifying changepoint probability.
            max_run_length: Maximum run length to track (for memory efficiency).
                           If None, no limit is imposed.

        Raises:
            ValueError: If max_run_length is not positive.
        """
        if max_run_length is not None and max_run_length <= 0:
            raise ValueError(f"max_run_length must be positive, got {max_run_length}")

        self.base_model = model
        self.hazard = hazard
        self.max_run_length = max_run_length

        # Internal state
        self.run_length_dist: Optional[np.ndarray] = None
        self.models: Optional[list[PredictiveModel]] = None
        self.timestep: int = 0

    def fit(self, data: np.ndarray) -> "BOCPD":
        """Initialize model parameters from historical data.

        Uses empirical Bayes to estimate hyperparameters and sets up
        the initial state for online detection.

        Args:
            data: Historical observations (1D array).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If data is empty.
        """
        if len(data) == 0:
            raise ValueError("Data must not be empty")

        # Initialize model hyperparameters using empirical Bayes
        self.base_model.fit_empirical(data)

        # Initialize state: start with run length 0 (certainty of changepoint at t=0)
        self.run_length_dist = np.array([0.0])  # log probability
        self.models = [self.base_model.copy()]
        self.timestep = 0

        return self

    def update(self, x: float) -> dict:
        """Update the detector with a new observation.

        Performs one step of the BOCPD algorithm:
        1. Compute predictive probabilities for each run length
        2. Calculate growth and changepoint probabilities
        3. Update run length distribution
        4. Update models for each run length

        Args:
            x: New observation.

        Returns:
            Dictionary containing:
                - 'run_length_dist': Current run length distribution (probabilities, not log)
                - 'changepoint_prob': Probability of changepoint at this timestep
                - 'most_likely_run_length': Most probable run length
                - 'prediction_log_prob': Log probability of the observation

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.run_length_dist is None or self.models is None:
            raise RuntimeError("Must call fit() before update()")

        # Save previous run length distribution for prediction log prob calculation
        prev_run_length_dist = self.run_length_dist

        # Step 1: Compute predictive probabilities
        # p(x_t | r_{t-1}, x_{1:t-1}) for each run length r_{t-1}
        pred_log_probs = np.array([model.predict(x) for model in self.models])

        # Step 2: Compute growth probabilities (no changepoint)
        # p(r_t = r_{t-1} + 1 | x_{1:t}) \propto p(x_t | r_{t-1}) * (1 - h(r_{t-1})) * p(r_{t-1} | x_{1:t-1})
        log_growth_probs = (
            pred_log_probs
            + np.array([self.hazard.compute_log_survival(r) for r in range(len(self.models))])
            + prev_run_length_dist
        )

        # Step 3: Compute changepoint probability (r_t = 0)
        # p(r_t = 0 | x_{1:t}) \propto p(x_t | r_t=0) * sum_r h(r) * p(r_{t-1} = r | x_{1:t-1})
        log_changepoint_prob_terms = (
            pred_log_probs
            + np.array([self.hazard.compute_log(r) for r in range(len(self.models))])
            + prev_run_length_dist
        )
        log_changepoint_prob = self._log_sum_exp(log_changepoint_prob_terms)

        # Step 4: Combine to get new run length distribution
        # Prepend changepoint probability and append growth probabilities
        new_log_run_length_dist = np.concatenate(
            [[log_changepoint_prob], log_growth_probs]
        )

        # Step 5: Normalize
        new_log_run_length_dist -= self._log_sum_exp(new_log_run_length_dist)

        # Step 6: Truncate if max_run_length is set
        if self.max_run_length is not None and len(new_log_run_length_dist) > self.max_run_length:
            new_log_run_length_dist = new_log_run_length_dist[:self.max_run_length]
            # Renormalize after truncation
            new_log_run_length_dist -= self._log_sum_exp(new_log_run_length_dist)

        # Step 7: Update models
        # Add new model for run length 0 (post-changepoint)
        # Update existing models with the new observation
        new_models = [self.base_model.copy()]  # r=0: fresh model
        for model in self.models:
            new_models.append(model.update(x))

        # Truncate models to match run length distribution
        if len(new_models) > len(new_log_run_length_dist):
            new_models = new_models[:len(new_log_run_length_dist)]

        # Update state
        self.run_length_dist = new_log_run_length_dist
        self.models = new_models
        self.timestep += 1

        # Prepare return values
        run_length_probs = np.exp(self.run_length_dist)
        changepoint_prob = run_length_probs[0]  # P(r_t = 0)
        most_likely_run_length = int(np.argmax(self.run_length_dist))

        # Overall prediction log probability (marginalized over run lengths)
        prediction_log_prob = self._log_sum_exp(
            pred_log_probs + prev_run_length_dist
        )

        return {
            "run_length_dist": run_length_probs,
            "changepoint_prob": changepoint_prob,
            "most_likely_run_length": most_likely_run_length,
            "prediction_log_prob": prediction_log_prob,
        }

    def predict(self) -> tuple[float, float]:
        """Predict the next observation (mean and variance).

        Computes the posterior predictive distribution by marginalizing
        over the current run length distribution.

        Note: This is a simplified version that assumes Gaussian predictive
        distributions. For general models, this may need to be adapted.

        Returns:
            Tuple of (predicted_mean, predicted_variance).

        Raises:
            RuntimeError: If fit() has not been called.
            NotImplementedError: For non-Gaussian models.
        """
        if self.run_length_dist is None or self.models is None:
            raise RuntimeError("Must call fit() before predict()")

        # This is a placeholder - proper implementation would require
        # model-specific prediction methods
        raise NotImplementedError(
            "Prediction is model-specific. "
            "Access models directly or implement in subclass."
        )

    def get_run_length_distribution(self) -> np.ndarray:
        """Get the current run length distribution.

        Returns:
            Array of probabilities for each run length.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.run_length_dist is None:
            raise RuntimeError("Must call fit() before accessing run length distribution")

        return np.exp(self.run_length_dist)

    def get_changepoint_probability(self) -> float:
        """Get the probability of a changepoint at the current timestep.

        Returns:
            Probability that r_t = 0 (i.e., a changepoint just occurred).

        Raises:
            RuntimeError: If fit() has not been called or no updates performed.
        """
        if self.run_length_dist is None:
            raise RuntimeError("Must call fit() before accessing changepoint probability")
        if self.timestep == 0:
            raise RuntimeError("Must call update() at least once")

        return np.exp(self.run_length_dist[0])

    def get_most_likely_run_length(self) -> int:
        """Get the most likely run length.

        Returns:
            The run length with maximum posterior probability.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.run_length_dist is None:
            raise RuntimeError("Must call fit() before accessing run length")

        return int(np.argmax(self.run_length_dist))

    @staticmethod
    def _log_sum_exp(log_probs: np.ndarray) -> float:
        """Compute log(sum(exp(log_probs))) in a numerically stable way.

        Args:
            log_probs: Array of log probabilities.

        Returns:
            log(sum(exp(log_probs)))
        """
        max_log_prob = np.max(log_probs)
        if np.isinf(max_log_prob):
            return max_log_prob
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
