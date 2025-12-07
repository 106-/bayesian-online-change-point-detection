"""ベイズオンライン変化点検知のためのハザード関数基底クラス"""

from abc import ABC, abstractmethod


class HazardFunction(ABC):
    """ハザード関数の抽象基底クラス

    ハザード関数は、各ランレングス（最後の変化点からの経過時間）における
    変化点発生確率を指定します。異なるハザード関数は、ランレングスの分布に
    ついて異なる仮定をエンコードします。
    """

    @abstractmethod
    def compute(self, r: int) -> float:
        """ランレングスrでのハザード関数を計算

        ハザード関数h(r)は、現在のランレングスがrである場合に、
        時刻tで変化点が発生する確率を表します。

        Args:
            r: ランレングス（非負整数）

        Returns:
            [0, 1]の範囲のハザード確率h(r)

        Raises:
            ValueError: rが負の場合
        """
        pass

    def compute_survival(self, r: int) -> float:
        """ランレングスrでの生存関数を計算

        生存関数S(r) = 1 - h(r)は、変化点が発生しない確率を表します。
        このメソッドは効率化のためにオーバーライドできます。

        Args:
            r: ランレングス（非負整数）

        Returns:
            [0, 1]の範囲の生存確率S(r)
        """
        return 1.0 - self.compute(r)

    def compute_log(self, r: int) -> float:
        """ランレングスrでのハザード関数の対数を計算

        このメソッドは数値安定性のためにオーバーライドできます。

        Args:
            r: ランレングス（非負整数）

        Returns:
            ハザード確率の対数log(h(r))
        """
        import numpy as np

        hazard = self.compute(r)
        if hazard == 0:
            return -np.inf
        return np.log(hazard)

    def compute_log_survival(self, r: int) -> float:
        """ランレングスrでの生存関数の対数を計算

        このメソッドは数値安定性のためにオーバーライドできます。

        Args:
            r: ランレングス（非負整数）

        Returns:
            生存確率の対数log(S(r))
        """
        import numpy as np

        survival = self.compute_survival(r)
        if survival == 0:
            return -np.inf
        return np.log(survival)
