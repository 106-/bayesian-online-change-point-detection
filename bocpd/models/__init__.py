"""Predictive models for Bayesian online changepoint detection."""

from bocpd.models.base import PredictiveModel
from bocpd.models.gaussian import GaussianModel

__all__ = ["PredictiveModel", "GaussianModel"]
