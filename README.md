# BOCPD - オンラインベイズ変化点検知

オンラインベイズ変化点検知(BOCPD)を実装するものです。実験用途のもののため、間違いが含まれている可能性があります。

## インストール

```bash
# 基本インストール
uv pip install .

# 可視化機能を含む
uv pip install ".[viz]"

# 開発ツールを含む
uv pip install ".[dev]"
```

## クイックスタート

```python
import numpy as np
from bocpd import BOCPD, GaussianModel, ConstantHazard

# 過去データで初期化
historical_data = np.random.randn(100)
model = GaussianModel()
hazard = ConstantHazard(lambda_=0.01)  # 期待ランレングス: 100
detector = BOCPD(model=model, hazard=hazard)
detector.fit(historical_data)

# オンライン処理
for new_observation in data_stream:
    result = detector.update(new_observation)

    # 変化点検知
    if result["most_likely_run_length"] == 1 and result["run_length_dist"][1] > 0.5:
        print("変化点を検知しました！")
```

## アーキテクチャ

### コアコンポーネント

#### 1. `BOCPD` (detector.py)
メインの検知器クラス。ランレングス分布を維持し、逐次的に更新します。

**主要メソッド:**
- `fit(data)`: 過去データでパラメータを推定
- `update(x)`: 新しいデータで更新
- `get_run_length_distribution()`: 現在のランレングス分布を取得
- `get_changepoint_probability()`: 変化点確率を取得

**返り値 (update):**
```python
{
    "run_length_dist": np.ndarray,  # ランレングスの確率分布
    "changepoint_prob": float,       # P(r_t = 0)
    "most_likely_run_length": int,   # 最も確からしいランレングス
    "prediction_log_prob": float     # 予測対数確率
}
```

#### 2. `PredictiveModel` (models/base.py)
予測モデルの抽象基底クラス。

**必須メソッド:**
- `fit_empirical(data)`: 経験ベイズでハイパーパラメータを推定
- `predict(x)`: 予測分布の対数確率密度を計算
- `update(x)`: データで事後分布を更新（新しいインスタンスを返す）
- `copy()`: モデルのコピーを作成

**実装済みモデル:**
- `GaussianModel`: ガウス尤度、正規-逆ガンマ事前分布

#### 3. `HazardFunction` (hazards/base.py)
変化点発生確率をモデル化。

**必須メソッド:**
- `compute(r)`: ランレングスrに対するハザード関数の値

**実装済みハザード関数:**
- `ConstantHazard`: 一定のハザード率

## 使用例

demo.ipynbに基本的な使用例が載っています。

## API リファレンス

### BOCPD

```python
BOCPD(model: PredictiveModel, hazard: HazardFunction, max_run_length: Optional[int] = None)
```

**パラメータ:**
- `model`: 使用する予測モデル
- `hazard`: ハザード関数
- `max_run_length`: ランレングスの最大値（メモリ効率化）

### GaussianModel

```python
GaussianModel(mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0)
```

**パラメータ:**
- `mu0`: 平均の事前分布の平均
- `kappa0`: 平均の事前分布の精度スケーリング
- `alpha0`: 精度の事前分布の形状パラメータ
- `beta0`: 精度の事前分布のレートパラメータ

### ConstantHazard

```python
ConstantHazard(lambda_: float)
```

**パラメータ:**
- `lambda_`: ハザード率 (0 < lambda_ <= 1)
  - 期待ランレングス = 1 / lambda_
  - 例: lambda_=0.01 → 期待ランレングス100

## 参考文献

[Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. arXiv preprint arXiv:0710.3742.](https://arxiv.org/pdf/0710.3742)

