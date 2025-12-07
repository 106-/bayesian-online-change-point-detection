"""一定ハザード関数の実装"""

import numpy as np

from bocpd.hazards.base import HazardFunction


class ConstantHazard(HazardFunction):
    """一定ハザード関数

    ランレングスに関係なく、各時間ステップで一定の変化点発生確率を仮定します。
    これは、変化点間の時間の指数分布に対応します。

    lambda_がハザード率の場合、期待ランレングスは1/lambda_です。
    """

    def __init__(self, lambda_: float) -> None:
        """一定ハザード関数を初期化

        Args:
            lambda_: 一定のハザード率（0 < lambda_ <= 1）
                    例: lambda_=0.01は期待ランレングス100を意味する

        Raises:
            ValueError: lambda_が(0, 1]の範囲にない場合
        """
        if not (0 < lambda_ <= 1):
            raise ValueError(f"lambda_ must be in (0, 1], got {lambda_}")

        self.lambda_ = lambda_

    def compute(self, r: int) -> float:
        """ランレングスrでの一定ハザードを計算

        Args:
            r: ランレングス（非負整数）

        Returns:
            一定のハザード確率lambda_

        Raises:
            ValueError: rが負の場合
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return self.lambda_

    def compute_survival(self, r: int) -> float:
        """生存確率を計算

        Args:
            r: ランレングス（非負整数）

        Returns:
            生存確率 1 - lambda_
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return 1.0 - self.lambda_

    def compute_log(self, r: int) -> float:
        """ハザード関数の対数を計算

        Args:
            r: ランレングス（非負整数）

        Returns:
            ハザード確率の対数log(lambda_)
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return np.log(self.lambda_)

    def compute_log_survival(self, r: int) -> float:
        """生存関数の対数を計算

        Args:
            r: ランレングス（非負整数）

        Returns:
            生存確率の対数log(1 - lambda_)
        """
        if r < 0:
            raise ValueError(f"Run length must be non-negative, got {r}")

        return np.log1p(-self.lambda_)  # 数値安定性のためlog1pを使用

    def __repr__(self) -> str:
        """文字列表現"""
        expected_run_length = 1.0 / self.lambda_
        return f"ConstantHazard(lambda_={self.lambda_:.4f}, expected_run_length={expected_run_length:.2f})"
