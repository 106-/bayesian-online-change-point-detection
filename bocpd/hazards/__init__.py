"""Hazard functions for Bayesian online changepoint detection."""

from bocpd.hazards.base import HazardFunction
from bocpd.hazards.constant import ConstantHazard

__all__ = ["HazardFunction", "ConstantHazard"]
