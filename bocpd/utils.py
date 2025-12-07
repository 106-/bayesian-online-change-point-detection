"""ベイズオンライン変化点検知のためのユーティリティ関数"""

from typing import Optional

import numpy as np


def log_sum_exp(log_probs: np.ndarray) -> float:
    """数値的に安定した方法でlog(sum(exp(log_probs)))を計算

    これは、対数確率を扱う際の数値安定性のためのlog-sum-expトリックです。

    Args:
        log_probs: 対数確率の配列

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
    """ランレングス分布に基づいて変化点が発生したかを検知

    P(r_t = 0) > thresholdの場合に変化点と判定します。

    Args:
        run_length_dist: ランレングス確率分布（対数確率ではない）
        threshold: 変化点検知の閾値（デフォルト: 0.5）

    Returns:
        変化点が検知された場合True、それ以外False

    Raises:
        ValueError: thresholdが[0, 1]の範囲にない場合

    Example:
        >>> dist = np.array([0.7, 0.2, 0.1])  # r=0で高い確率
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
    """メモリ効率化のため、非常に低い確率のランレングスを刈り込む

    閾値以下の確率を持つランレングスを削除し、再正規化します。

    Args:
        log_run_length_dist: 対数ランレングス分布
        models: 各ランレングスに対応するモデルのリスト
        threshold: 刈り込みの確率閾値（デフォルト: 1e-10）

    Returns:
        (刈り込まれた対数分布, 刈り込まれたモデル)のタプル

    Example:
        >>> log_dist = np.array([0, -5, -20, -30])  # 最後の2つは非常に低確率
        >>> models = [f"model_{i}" for i in range(4)]
        >>> new_dist, new_models = prune_run_lengths(log_dist, models, threshold=1e-8)
        >>> len(new_models) < len(models)
        True
    """
    probs = np.exp(log_run_length_dist)
    mask = probs >= threshold

    if not np.any(mask):
        # 少なくとも最も確からしいものを1つ保持
        max_idx = np.argmax(log_run_length_dist)
        mask = np.zeros_like(mask)
        mask[max_idx] = True

    pruned_log_dist = log_run_length_dist[mask]
    pruned_models = [model for i, model in enumerate(models) if mask[i]]

    # 再正規化
    pruned_log_dist -= log_sum_exp(pruned_log_dist)

    return pruned_log_dist, pruned_models


def plot_run_length_history(
    history: list[np.ndarray],
    figsize: tuple[int, int] = (12, 6),
    title: str = "時間経過に伴うランレングス分布",
) -> "matplotlib.figure.Figure":
    """時間経過に伴うランレングス分布をプロット

    ランレングス分布の変化を示すヒートマップを作成します。
    変化点検知の可視化に有用です。

    Args:
        history: 各タイムステップでのランレングス分布（確率、対数ではない）のリスト
        figsize: 図のサイズ（幅、高さ）インチ単位
        title: プロットのタイトル

    Returns:
        MatplotlibのFigureオブジェクト

    Raises:
        ImportError: matplotlibがインストールされていない場合

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

    # 2次元配列に変換（短い分布はゼロでパディング）
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

    ax.set_xlabel("時刻")
    ax.set_ylabel("ランレングス")
    ax.set_title(title)

    # カラーバーを追加
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("確率")

    # 変化点を強調表示（r=0が高確率の場所）
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
    """観測値の負の対数尤度を計算

    モデルの予測性能を評価するために使用できます。

    Args:
        observations: 観測値の配列
        prediction_log_probs: 各観測値の対数確率の配列

    Returns:
        負の対数尤度（小さいほど良い）

    Example:
        >>> obs = np.array([1.0, 2.0, 3.0])
        >>> log_probs = np.array([-0.5, -0.6, -0.4])
        >>> nll = compute_negative_log_likelihood(obs, log_probs)
        >>> nll > 0
        True
    """
    return -np.sum(prediction_log_probs)
