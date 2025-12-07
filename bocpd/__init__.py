"""ベイズオンライン変化点検知（BOCPD）ライブラリ

このライブラリは、様々な共役モデルとハザード関数をサポートする
ベイズオンライン変化点検知の実用的な実装を提供します。

Example:
    >>> import numpy as np
    >>> from bocpd import BOCPD, GaussianModel, ConstantHazard
    >>>
    >>> # モデルと検知器を初期化
    >>> model = GaussianModel()
    >>> hazard = ConstantHazard(lambda_=0.01)  # 期待ランレングス100
    >>> detector = BOCPD(model=model, hazard=hazard)
    >>>
    >>> # 過去データで学習
    >>> historical_data = np.random.randn(100)
    >>> detector.fit(historical_data)
    >>>
    >>> # 新しい観測値を処理
    >>> for x in np.random.randn(50):
    ...     result = detector.update(x)
    ...     if result["changepoint_prob"] > 0.5:
    ...         print(f"変化点を検知！")
"""

from bocpd.detector import BOCPD
from bocpd.hazards import ConstantHazard, HazardFunction
from bocpd.models import GaussianModel, PredictiveModel

__version__ = "0.1.0"

__all__ = [
    "BOCPD",
    "PredictiveModel",
    "GaussianModel",
    "HazardFunction",
    "ConstantHazard",
]
