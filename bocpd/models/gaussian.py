"""正規-逆ガンマ事前分布を持つガウス予測モデル"""

from typing import Any

import numpy as np
from scipy import stats

from bocpd.models.base import PredictiveModel


class GaussianModel(PredictiveModel):
    """正規-逆ガンマ共役事前分布を持つガウス尤度モデル

    このモデルは、観測値が未知の平均と分散を持つガウス分布から生成されると仮定します。
    事前分布は正規-逆ガンマ分布であり、共役性を維持し、解析的な事後更新を可能にします。

    ハイパーパラメータ:
        mu0: ガウス平均パラメータの事前平均
        kappa0: 平均の事前精度（逆分散）スケーリング
        alpha0: 精度（逆分散）の事前形状パラメータ
        beta0: 精度の事前レートパラメータ

    予測分布は、ハイパーパラメータから導出されたパラメータを持つスチューデントのt分布です。
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        """ハイパーパラメータでガウスモデルを初期化

        Args:
            mu0: 事前平均（デフォルト: 0.0）
            kappa0: 事前精度スケーリング（デフォルト: 1.0、正の値である必要あり）
            alpha0: 事前形状パラメータ（デフォルト: 1.0、正の値である必要あり）
            beta0: 事前レートパラメータ（デフォルト: 1.0、正の値である必要あり）

        Raises:
            ValueError: kappa0、alpha0、beta0が正でない場合
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
        """経験ベイズを使用してハイパーパラメータを初期化

        サンプル統計量を使用してデータからハイパーパラメータを推定します。
        経験的推定値を中心とした弱情報事前分布を設定します。

        Args:
            data: 過去観測データ（1次元配列）

        Raises:
            ValueError: データが空または2未満の観測値しか含まない場合
        """
        if len(data) == 0:
            raise ValueError("Data must not be empty")
        if len(data) < 2:
            raise ValueError("Need at least 2 observations for empirical fitting")

        # サンプル統計量を計算
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # mu0をサンプル平均に設定
        self.mu0 = sample_mean

        # kappa0を弱情報的に設定（約1観測値相当）
        self.kappa0 = 1.0

        # サンプル分散に基づいてalpha0とbeta0を設定
        # モーメント法を使用: E[分散] = beta/(alpha-1) (alpha > 1の場合)
        # alpha0 = 2（有限平均の最小値）に設定し、それに応じてbeta0を設定
        self.alpha0 = 2.0
        self.beta0 = sample_var * (self.alpha0 - 1)

    def predict(self, x: float) -> float:
        """予測分布の下での対数確率密度を計算

        予測分布は以下のパラメータを持つスチューデントのt分布です:
        - 自由度: 2 * alpha0
        - 位置: mu0
        - スケール: sqrt(beta0 * (kappa0 + 1) / (alpha0 * kappa0))

        Args:
            x: 評価する観測値

        Returns:
            対数確率密度 log p(x | ハイパーパラメータ)
        """
        # スチューデントのtパラメータ
        df = 2 * self.alpha0
        loc = self.mu0
        scale = np.sqrt(self.beta0 * (self.kappa0 + 1) / (self.alpha0 * self.kappa0))

        # scipyを使用して対数確率を計算
        log_prob = stats.t.logpdf(x, df=df, loc=loc, scale=scale)

        return log_prob

    def update(self, x: float) -> "GaussianModel":
        """新しい観測値が与えられた場合の事後分布を更新

        共役性を利用して解析的なベイズ更新を実行します。

        Args:
            x: 新しい観測値

        Returns:
            更新されたハイパーパラメータを持つ新しいGaussianModelインスタンス
        """
        # 事後ハイパーパラメータ（共役更新）
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
        """モデルの深いコピーを作成

        Returns:
            同じハイパーパラメータを持つ新しいGaussianModelインスタンス
        """
        return GaussianModel(
            mu0=self.mu0,
            kappa0=self.kappa0,
            alpha0=self.alpha0,
            beta0=self.beta0,
        )

    def get_params(self) -> dict[str, Any]:
        """現在のハイパーパラメータを取得

        Returns:
            キー: mu0, kappa0, alpha0, beta0を持つ辞書
        """
        return {
            "mu0": self.mu0,
            "kappa0": self.kappa0,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
        }

    def set_params(self, **params: Any) -> None:
        """ハイパーパラメータを設定

        Args:
            **params: 設定するハイパーパラメータ（mu0, kappa0, alpha0, beta0）

        Raises:
            ValueError: パラメータが無効な場合
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
        """モデルの文字列表現"""
        return (
            f"GaussianModel(mu0={self.mu0:.4f}, kappa0={self.kappa0:.4f}, "
            f"alpha0={self.alpha0:.4f}, beta0={self.beta0:.4f})"
        )
