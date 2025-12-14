"""ベイズオンライン変化点検知（BOCPD）の実装"""

from typing import Optional

import numpy as np

from bocpd.hazards.base import HazardFunction
from bocpd.models.base import PredictiveModel


class BOCPD:
    """ベイズオンライン変化点検知

    このクラスは、逐次データ中の変化点を検知するためのBOCPDアルゴリズムを実装します。
    ランレングス（最後の変化点からの経過時間）の分布を維持し、新しい観測値が到着
    するたびにオンラインで更新します。

    参考文献:
        Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
        arXiv preprint arXiv:0710.3742.
    """

    def __init__(
        self,
        model: PredictiveModel,
        hazard: HazardFunction,
        max_run_length: Optional[int] = None,
    ) -> None:
        """BOCPD検知器を初期化

        Args:
            model: 予測モデル（各ランレングスごとにコピーされます）
            hazard: 変化点確率を指定するハザード関数
            max_run_length: 追跡する最大ランレングス（メモリ効率化のため）
                           Noneの場合、制限なし

        Raises:
            ValueError: max_run_lengthが正でない場合
        """
        if max_run_length is not None and max_run_length <= 0:
            raise ValueError(f"max_run_length must be positive, got {max_run_length}")

        self.base_model = model
        self.hazard = hazard
        self.max_run_length = max_run_length

        # 内部状態
        self.run_length_dist: Optional[np.ndarray] = None
        self.models: Optional[list[PredictiveModel]] = None
        self.timestep: int = 0

    def fit(self, data: np.ndarray) -> "BOCPD":
        """過去データからモデルパラメータを初期化

        経験ベイズを使用してハイパーパラメータを推定し、
        オンライン検知のための初期状態を設定します。

        Args:
            data: 過去観測データ（1次元配列）

        Returns:
            メソッドチェーンのためのself

        Raises:
            ValueError: データが空の場合
        """
        if len(data) == 0:
            raise ValueError("Data must not be empty")

        # 経験ベイズを使用してモデルのハイパーパラメータを初期化
        self.base_model.fit_empirical(data)

        # 状態を初期化: ランレングス0で開始（t=0で変化点が確実に発生）
        self.run_length_dist = np.array([0.0])  # 対数確率
        self.models = [self.base_model.copy()]
        self.timestep = 0

        return self

    def update(self, x: float) -> dict:
        """新しい観測値で検知器を更新

        BOCPDアルゴリズムの1ステップを実行:
        1. 各ランレングスの予測確率を計算
        2. 成長確率と変化点確率を計算
        3. ランレングス分布を更新
        4. 各ランレングスのモデルを更新

        Args:
            x: 新しい観測値

        Returns:
            以下を含む辞書:
                - 'run_length_dist': 現在のランレングス分布（確率、対数ではない）
                - 'changepoint_prob': このタイムステップでの変化点確率
                - 'most_likely_run_length': 最も確からしいランレングス
                - 'prediction_log_prob': 観測値の対数確率

        Raises:
            RuntimeError: fit()が呼ばれていない場合
        """
        if self.run_length_dist is None or self.models is None:
            raise RuntimeError("Must call fit() before update()")

        # 予測対数確率計算のために前のランレングス分布を保存
        prev_run_length_dist = self.run_length_dist

        # ステップ1: 予測確率を計算
        # 各ランレングス r_{t-1} について p(x_t | r_{t-1}, x_{1:t-1})
        pred_log_probs = np.array([model.predict(x) for model in self.models])

        # ステップ2: 成長確率を計算（変化点なし）
        # p(r_t = r_{t-1} + 1 | x_{1:t}) \propto p(x_t | r_{t-1}) * (1 - h(r_{t-1})) * p(r_{t-1} | x_{1:t-1})
        log_growth_probs = (
            pred_log_probs
            + np.array([self.hazard.compute_log_survival(r) for r in range(len(self.models))])
            + prev_run_length_dist
        )

        # ステップ3: 変化点確率を計算（r_t = 0）
        # p(r_t = 0 | x_{1:t}) \propto p(x_t | r_t=0) * sum_r h(r) * p(r_{t-1} = r | x_{1:t-1})
        log_changepoint_prob_terms = (
            self.base_model.predict(x)
            + np.array([self.hazard.compute_log(r) for r in range(len(self.models))])
            + prev_run_length_dist
        )
        log_changepoint_prob = self._log_sum_exp(log_changepoint_prob_terms)

        # ステップ4: 結合して新しいランレングス分布を取得
        # 変化点確率を先頭に、成長確率を追加
        new_log_run_length_dist = np.concatenate(
            [[log_changepoint_prob], log_growth_probs]
        )

        # ステップ5: 正規化
        new_log_run_length_dist -= self._log_sum_exp(new_log_run_length_dist)

        # ステップ6: max_run_lengthが設定されている場合は切り詰め
        if self.max_run_length is not None and len(new_log_run_length_dist) > self.max_run_length:
            new_log_run_length_dist = new_log_run_length_dist[:self.max_run_length]
            # 切り詰め後に再正規化
            new_log_run_length_dist -= self._log_sum_exp(new_log_run_length_dist)

        # ステップ7: モデルを更新
        # ランレングス0用の新しいモデルを追加（変化点後）
        # 既存のモデルを新しい観測値で更新
        new_models = [self.base_model.copy()]  # r=0: 新鮮なモデル
        for model in self.models:
            new_models.append(model.update(x))

        # ランレングス分布に合わせてモデルを切り詰め
        if len(new_models) > len(new_log_run_length_dist):
            new_models = new_models[:len(new_log_run_length_dist)]

        # 状態を更新
        self.run_length_dist = new_log_run_length_dist
        self.models = new_models
        self.timestep += 1

        # 返り値を準備
        run_length_probs = np.exp(self.run_length_dist)
        changepoint_prob = run_length_probs[0]  # P(r_t = 0)
        most_likely_run_length = int(np.argmax(self.run_length_dist))

        # 全体の予測対数確率（ランレングスで周辺化）
        prediction_log_prob = self._log_sum_exp(
            pred_log_probs + prev_run_length_dist
        )

        return {
            "run_length_dist": run_length_probs,
            "changepoint_prob": changepoint_prob,
            "most_likely_run_length": most_likely_run_length,
            "prediction_log_prob": prediction_log_prob,
        }

    def predict(self) -> tuple[float, float]:
        """次の観測値を予測（平均と分散）

        現在のランレングス分布で周辺化して事後予測分布を計算します。

        注意: これは簡易版で、ガウス予測分布を仮定しています。
        一般的なモデルの場合、適応が必要な場合があります。

        Returns:
            (予測平均, 予測分散)のタプル

        Raises:
            RuntimeError: fit()が呼ばれていない場合
            NotImplementedError: 非ガウスモデルの場合
        """
        if self.run_length_dist is None or self.models is None:
            raise RuntimeError("Must call fit() before predict()")

        # これはプレースホルダー - 適切な実装にはモデル固有の予測メソッドが必要
        raise NotImplementedError(
            "Prediction is model-specific. "
            "Access models directly or implement in subclass."
        )

    def get_run_length_distribution(self) -> np.ndarray:
        """現在のランレングス分布を取得

        Returns:
            各ランレングスの確率の配列

        Raises:
            RuntimeError: fit()が呼ばれていない場合
        """
        if self.run_length_dist is None:
            raise RuntimeError("Must call fit() before accessing run length distribution")

        return np.exp(self.run_length_dist)

    def get_changepoint_probability(self) -> float:
        """現在のタイムステップでの変化点確率を取得

        Returns:
            r_t = 0の確率（つまり、変化点が発生したばかり）

        Raises:
            RuntimeError: fit()が呼ばれていない、または更新が実行されていない場合
        """
        if self.run_length_dist is None:
            raise RuntimeError("Must call fit() before accessing changepoint probability")
        if self.timestep == 0:
            raise RuntimeError("Must call update() at least once")

        return np.exp(self.run_length_dist[0])

    def get_most_likely_run_length(self) -> int:
        """最も確からしいランレングスを取得

        Returns:
            事後確率が最大のランレングス

        Raises:
            RuntimeError: fit()が呼ばれていない場合
        """
        if self.run_length_dist is None:
            raise RuntimeError("Must call fit() before accessing run length")

        return int(np.argmax(self.run_length_dist))

    @staticmethod
    def _log_sum_exp(log_probs: np.ndarray) -> float:
        """数値的に安定した方法でlog(sum(exp(log_probs)))を計算

        Args:
            log_probs: 対数確率の配列

        Returns:
            log(sum(exp(log_probs)))
        """
        max_log_prob = np.max(log_probs)
        if np.isinf(max_log_prob):
            return max_log_prob
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
