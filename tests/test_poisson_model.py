"""ポアソンモデルのテスト"""

import numpy as np
import pytest

from bocpd.models import PoissonModel


class TestPoissonModel:
    """PoissonModelクラスのテスト"""

    def test_initialization(self):
        """デフォルトパラメータでの初期化をテスト"""
        model = PoissonModel()
        assert model.alpha == 1.0
        assert model.beta == 1.0

    def test_initialization_with_params(self):
        """カスタムパラメータでの初期化をテスト"""
        model = PoissonModel(alpha=2.0, beta=3.0)
        assert model.alpha == 2.0
        assert model.beta == 3.0

    def test_initialization_invalid_alpha(self):
        """alphaが負の場合にValueErrorが発生することをテスト"""
        with pytest.raises(ValueError, match="alpha must be positive"):
            PoissonModel(alpha=-1.0, beta=1.0)

    def test_initialization_invalid_beta(self):
        """betaが負の場合にValueErrorが発生することをテスト"""
        with pytest.raises(ValueError, match="beta must be positive"):
            PoissonModel(alpha=1.0, beta=-1.0)

    def test_fit_empirical(self):
        """経験ベイズ初期化をテスト"""
        # 既知の平均を持つポアソンデータを生成
        np.random.seed(42)
        true_rate = 5.0
        data = np.random.poisson(true_rate, size=1000)

        model = PoissonModel()
        model.fit_empirical(data)

        # alphaは小さい値（弱情報事前分布）に設定されるべき
        assert model.alpha == 2.0
        # betaはサンプル平均に基づいて設定されるべき
        # E[lambda] = alpha / beta ≈ sample_mean
        expected_mean = model.alpha / model.beta
        assert abs(expected_mean - np.mean(data)) < 0.5

    def test_fit_empirical_empty_data(self):
        """空のデータでValueErrorが発生することをテスト"""
        model = PoissonModel()
        with pytest.raises(ValueError, match="Data must not be empty"):
            model.fit_empirical(np.array([]))

    def test_predict(self):
        """予測関数が対数確率を返すことをテスト"""
        model = PoissonModel(alpha=3.0, beta=2.0)

        # いくつかのカウント値で予測をテスト
        log_prob_0 = model.predict(0)
        log_prob_5 = model.predict(5)

        # 対数確率は0以下である必要がある
        assert log_prob_0 <= 0
        assert log_prob_5 <= 0

        # 負の値は-infを返すべき
        assert model.predict(-1) == -np.inf

    def test_update(self):
        """更新関数が新しいインスタンスを返すことをテスト"""
        model = PoissonModel(alpha=2.0, beta=1.0)
        x = 3

        # 更新
        updated_model = model.update(x)

        # 元のモデルは変更されていないことを確認（不変性）
        assert model.alpha == 2.0
        assert model.beta == 1.0

        # 更新されたモデルは正しいパラメータを持つべき
        assert updated_model.alpha == 2.0 + x
        assert updated_model.beta == 1.0 + 1

    def test_update_sequence(self):
        """一連の更新が正しく動作することをテスト"""
        model = PoissonModel(alpha=1.0, beta=1.0)
        observations = [2, 3, 1, 4]

        # 逐次更新
        for obs in observations:
            model = model.update(obs)

        # 最終的なパラメータを確認
        expected_alpha = 1.0 + sum(observations)
        expected_beta = 1.0 + len(observations)

        assert model.alpha == expected_alpha
        assert model.beta == expected_beta

    def test_copy(self):
        """コピー関数が独立したインスタンスを作成することをテスト"""
        model = PoissonModel(alpha=2.0, beta=3.0)
        copied_model = model.copy()

        # パラメータが同じことを確認
        assert copied_model.alpha == model.alpha
        assert copied_model.beta == model.beta

        # 独立したインスタンスであることを確認
        assert copied_model is not model

    def test_get_params(self):
        """get_paramsが正しいパラメータを返すことをテスト"""
        model = PoissonModel(alpha=2.0, beta=3.0)
        params = model.get_params()

        assert params == {"alpha": 2.0, "beta": 3.0}

    def test_set_params(self):
        """set_paramsが正しく動作することをテスト"""
        model = PoissonModel()

        model.set_params(alpha=5.0, beta=7.0)
        assert model.alpha == 5.0
        assert model.beta == 7.0

        # 部分的な更新
        model.set_params(alpha=10.0)
        assert model.alpha == 10.0
        assert model.beta == 7.0

    def test_set_params_invalid(self):
        """set_paramsで無効なパラメータに対してValueErrorが発生することをテスト"""
        model = PoissonModel()

        with pytest.raises(ValueError, match="alpha must be positive"):
            model.set_params(alpha=-1.0)

        with pytest.raises(ValueError, match="beta must be positive"):
            model.set_params(beta=0.0)

    def test_repr(self):
        """文字列表現が正しいことをテスト"""
        model = PoissonModel(alpha=2.5, beta=3.5)
        repr_str = repr(model)

        assert "PoissonModel" in repr_str
        assert "alpha=2.5" in repr_str
        assert "beta=3.5" in repr_str

    def test_predictive_distribution_properties(self):
        """予測分布（負の二項分布）の性質をテスト"""
        model = PoissonModel(alpha=5.0, beta=2.0)

        # 複数の値で対数確率を計算
        x_values = np.arange(0, 20)
        log_probs = np.array([model.predict(x) for x in x_values])
        probs = np.exp(log_probs)

        # 確率の合計は約1になるべき（離散分布）
        # 注意: 無限の範囲なので、0-19の範囲では1未満
        assert 0.8 < np.sum(probs) < 1.0

        # 全ての確率は0と1の間
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
