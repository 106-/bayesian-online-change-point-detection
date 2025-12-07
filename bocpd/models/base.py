"""ベイズオンライン変化点検知のための予測モデル基底クラス"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PredictiveModel(ABC):
    """ベイズ共役予測モデルの抽象基底クラス

    BOCPDで使用するには、全ての予測モデルがこのインターフェースを実装する必要があります。
    モデルは共役事前分布のハイパーパラメータを維持し、経験ベイズ初期化、予測、更新の
    メソッドを提供する必要があります。
    """

    @abstractmethod
    def fit_empirical(self, data: np.ndarray) -> None:
        """過去データから経験ベイズでハイパーパラメータを初期化

        このメソッドは、観測データから妥当なハイパーパラメータを推定します。
        通常、サンプル統計量を計算し、弱情報事前分布を設定します。

        Args:
            data: 1次元numpy配列としての過去観測データ

        Raises:
            ValueError: データが空または無効な場合
        """
        pass

    @abstractmethod
    def predict(self, x: float) -> float:
        """予測分布の下での観測値xの対数確率密度を計算

        予測分布は、現在のハイパーパラメータが与えられた事後予測分布です
        （つまり、潜在パラメータを積分消去したもの）。

        Args:
            x: 評価する観測値

        Returns:
            対数確率密度 log p(x | D_t)。D_tはこれまでに観測されたデータ
        """
        pass

    @abstractmethod
    def update(self, x: float) -> "PredictiveModel":
        """新しい観測値が与えられた場合の事後分布を更新

        このメソッドは共役性を利用してベイズ更新を実行し、
        事後ハイパーパラメータを解析的に計算します。

        重要: このメソッドは更新されたパラメータを持つ新しいインスタンスを返し、
        元のインスタンスは変更しません（不変性）。これは、BOCPDが各ランレングス
        仮説ごとに独立したモデル状態を維持する必要があるためです。

        Args:
            x: 新しい観測値

        Returns:
            更新されたハイパーパラメータを持つ新しいPredictiveModelインスタンス
        """
        pass

    @abstractmethod
    def copy(self) -> "PredictiveModel":
        """モデルの深いコピーを作成

        Returns:
            同じハイパーパラメータを持つ新しいPredictiveModelインスタンス
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """現在のハイパーパラメータを辞書として取得

        Returns:
            全てのハイパーパラメータを含む辞書
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> None:
        """辞書からハイパーパラメータを設定

        Args:
            **params: 設定するハイパーパラメータ

        Raises:
            ValueError: パラメータが無効な場合
        """
        pass
