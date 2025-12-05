"""Utility functions for Bayesian online changepoint detection."""

from typing import Optional

import numpy as np


def log_sum_exp(log_probs: np.ndarray) -> float:
    """Compute log(sum(exp(log_probs))) in a numerically stable way.

    This is the log-sum-exp trick for numerical stability when working
    with log probabilities.

    Args:
        log_probs: Array of log probabilities.

    Returns:
        log(sum(exp(log_probs)))

    Example:
        >>> log_probs = np.array([-1000, -1001, -1002])
        >>> result = log_sum_exp(log_probs)
        >>> np.isclose(result, -1000.0)
        True
    """
    max_log_prob = np.max(log_probs)
    if np.isinf(max_log_prob):
        return max_log_prob
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))


def detect_changepoints(
    run_length_dist: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    """Detect if a changepoint occurred based on the run length distribution.

    A changepoint is detected if P(r_t = 0) > threshold.

    Args:
        run_length_dist: Run length probability distribution (not log probabilities).
        threshold: Threshold for changepoint detection (default: 0.5).

    Returns:
        True if a changepoint is detected, False otherwise.

    Raises:
        ValueError: If threshold is not in [0, 1].

    Example:
        >>> dist = np.array([0.7, 0.2, 0.1])  # High probability at r=0
        >>> detect_changepoints(dist, threshold=0.5)
        True
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    return run_length_dist[0] > threshold


def prune_run_lengths(
    log_run_length_dist: np.ndarray,
    models: list,
    threshold: float = 1e-10,
) -> tuple[np.ndarray, list]:
    """Prune run lengths with very low probability for memory efficiency.

    Removes run lengths with probability below the threshold and renormalizes.

    Args:
        log_run_length_dist: Log run length distribution.
        models: List of models corresponding to each run length.
        threshold: Probability threshold for pruning (default: 1e-10).

    Returns:
        Tuple of (pruned_log_distribution, pruned_models).

    Example:
        >>> log_dist = np.array([0, -5, -20, -30])  # Last two are very unlikely
        >>> models = [f"model_{i}" for i in range(4)]
        >>> new_dist, new_models = prune_run_lengths(log_dist, models, threshold=1e-8)
        >>> len(new_models) < len(models)
        True
    """
    probs = np.exp(log_run_length_dist)
    mask = probs >= threshold

    if not np.any(mask):
        # Keep at least the most likely one
        max_idx = np.argmax(log_run_length_dist)
        mask = np.zeros_like(mask)
        mask[max_idx] = True

    pruned_log_dist = log_run_length_dist[mask]
    pruned_models = [model for i, model in enumerate(models) if mask[i]]

    # Renormalize
    pruned_log_dist -= log_sum_exp(pruned_log_dist)

    return pruned_log_dist, pruned_models


def plot_run_length_history(
    history: list[np.ndarray],
    figsize: tuple[int, int] = (12, 6),
    title: str = "Run Length Distribution Over Time",
) -> "matplotlib.figure.Figure":
    """Plot the run length distribution over time.

    Creates a heatmap showing how the run length distribution evolves,
    useful for visualizing changepoint detection.

    Args:
        history: List of run length distributions (probabilities, not log) at each timestep.
        figsize: Figure size (width, height) in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.

    Example:
        >>> history = [np.array([0.9, 0.1]), np.array([0.2, 0.7, 0.1])]
        >>> fig = plot_run_length_history(history)  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e

    # Convert to 2D array (pad shorter distributions with zeros)
    max_len = max(len(dist) for dist in history)
    matrix = np.zeros((len(history), max_len))
    for i, dist in enumerate(history):
        matrix[i, :len(dist)] = dist

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix.T,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Run Length")
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    # Highlight changepoints (where r=0 has high probability)
    changepoint_threshold = 0.5
    changepoints = [i for i, dist in enumerate(history) if dist[0] > changepoint_threshold]
    for cp in changepoints:
        ax.axvline(cp, color="blue", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    return fig


def compute_negative_log_likelihood(
    observations: np.ndarray,
    prediction_log_probs: np.ndarray,
) -> float:
    """Compute the negative log-likelihood of observations.

    This can be used to evaluate the model's predictive performance.

    Args:
        observations: Array of observed values.
        prediction_log_probs: Array of log probabilities for each observation.

    Returns:
        Negative log-likelihood (lower is better).

    Example:
        >>> obs = np.array([1.0, 2.0, 3.0])
        >>> log_probs = np.array([-0.5, -0.6, -0.4])
        >>> nll = compute_negative_log_likelihood(obs, log_probs)
        >>> nll > 0
        True
    """
    return -np.sum(prediction_log_probs)
