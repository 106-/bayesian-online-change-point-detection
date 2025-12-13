"""ガンマ事前分布を持つポアソン予測モデル"""

from typing import Any

import numpy as np
from scipy import stats
from scipy.special import gammaln

from bocpd.models.base import PredictiveModel


class PoissonModel(PredictiveModel):
    """ガンマ共役事前分布を持つポアソン尤度モデル

    このモデルは、観測値（非負整数カウント）が未知のレートパラメータを持つ
    ポアソン分布から生成されると仮定します。事前分布はガンマ分布であり、
    共役性を維持し、解析的な事後更新を可能にします。

    ハイパーパラメータ:
        alpha: ガンマ事前分布の形状パラメータ
        beta: ガンマ事前分布のレートパラメータ

    予測分布は、パラメータ r=alpha と p=beta/(beta+1) の負の二項分布です。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        """ハイパーパラメータでポアソンモデルを初期化

        Args:
            alpha: 事前形状パラメータ（デフォルト: 1.0、正の値である必要あり）
            beta: 事前レートパラメータ（デフォルト: 1.0、正の値である必要あり）

        Raises:
            ValueError: alphaまたはbetaが正でない場合
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        self.alpha = alpha
        self.beta = beta

    def fit_empirical(self, data: np.ndarray) -> None:
        """経験ベイズを使用してハイパーパラメータを初期化

        サンプル統計量を使用してデータからハイパーパラメータを推定します。
        ポアソン分布では平均=分散なので、サンプル平均を使用します。

        Args:
            data: 過去観測データ（1次元配列、非負整数を想定）

        Raises:
            ValueError: データが空の場合
        """
        if len(data) == 0:
            raise ValueError("Data must not be empty")

        # サンプル平均を計算
        sample_mean = np.mean(data)

        # モーメント法を使用: E[lambda] = alpha / beta
        # 弱情報事前分布として alpha を小さな値に設定
        self.alpha = 2.0
        self.beta = self.alpha / sample_mean if sample_mean > 0 else 1.0

    def predict(self, x: float) -> float:
        """予測分布の下での対数確率密度を計算

        予測分布は負の二項分布（NB分布）です:
        - パラメータ r (成功回数): alpha
        - パラメータ p (成功確率): beta / (beta + 1)

        Args:
            x: 評価する観測値（非負整数を想定）

        Returns:
            対数確率密度 log p(x | ハイパーパラメータ)
        """
        # xは整数であることを確認（カウントデータ）
        if x < 0:
            return -np.inf

        # 負の二項分布のパラメータ
        r = self.alpha
        p = self.beta / (self.beta + 1)

        # scipyの負の二項分布を使用
        # scipy.stats.nbinomは「失敗回数がxになるまでのr回の成功」を表す
        # したがって、パラメータはn=r, p=pです
        log_prob = stats.nbinom.logpmf(int(x), n=r, p=p)

        return log_prob

    def update(self, x: float) -> "PoissonModel":
        """新しい観測値が与えられた場合の事後分布を更新

        共役性を利用して解析的なベイズ更新を実行します。
        ポアソン-ガンマ共役更新式:
        - alpha_new = alpha + x
        - beta_new = beta + 1

        Args:
            x: 新しい観測値（非負整数を想定）

        Returns:
            更新されたハイパーパラメータを持つ新しいPoissonModelインスタンス
        """
        # 事後ハイパーパラメータ（共役更新）
        alpha_new = self.alpha + x
        beta_new = self.beta + 1

        return PoissonModel(
            alpha=alpha_new,
            beta=beta_new,
        )

    def copy(self) -> "PoissonModel":
        """モデルの深いコピーを作成

        Returns:
            同じハイパーパラメータを持つ新しいPoissonModelインスタンス
        """
        return PoissonModel(
            alpha=self.alpha,
            beta=self.beta,
        )

    def get_params(self) -> dict[str, Any]:
        """現在のハイパーパラメータを取得

        Returns:
            キー: alpha, betaを持つ辞書
        """
        return {
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def set_params(self, **params: Any) -> None:
        """ハイパーパラメータを設定

        Args:
            **params: 設定するハイパーパラメータ（alpha, beta）

        Raises:
            ValueError: パラメータが無効な場合
        """
        if "alpha" in params:
            if params["alpha"] <= 0:
                raise ValueError("alpha must be positive")
            self.alpha = params["alpha"]
        if "beta" in params:
            if params["beta"] <= 0:
                raise ValueError("beta must be positive")
            self.beta = params["beta"]

    def __repr__(self) -> str:
        """モデルの文字列表現"""
        return f"PoissonModel(alpha={self.alpha:.4f}, beta={self.beta:.4f})"
