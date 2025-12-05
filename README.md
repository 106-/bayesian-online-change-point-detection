# BOCPD - ベイズオンライン変化点検知

時系列データの変化点をリアルタイムで検知するための実用的なライブラリ

## 概要

BOCPDは、ストリーミング時系列データ中の変化点をリアルタイムで検知するためのPythonライブラリです。ベイズ共役モデルを利用した解析的な更新により、効率的かつ正確な変化点検知を実現します。

**主な特徴:**

- **経験ベイズによる初期化**: 過去データから自動的にパラメータを推定
- **逐次更新**: 一件ずつデータを処理し、リアルタイムで変化点を検知
- **拡張可能な設計**: ベイズ共役なモデルを簡単に追加可能
- **型安全**: 型ヒントによる明確なインターフェース

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

## デモ

```bash
uv run python main.py
```

シミュレートされたデータで変化点検知のデモを実行します。

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

### 基本的な使用方法

```python
import numpy as np
from bocpd import BOCPD, GaussianModel, ConstantHazard

# データ生成（3つのセグメント）
segment1 = np.random.normal(0, 1, 100)
segment2 = np.random.normal(5, 1, 100)
segment3 = np.random.normal(-3, 2, 100)
data = np.concatenate([segment1, segment2, segment3])

# 初期化
model = GaussianModel()
hazard = ConstantHazard(lambda_=0.01)
detector = BOCPD(model=model, hazard=hazard)

# 最初の50点で学習
detector.fit(data[:50])

# 残りのデータで逐次検知
for t, x in enumerate(data[50:], start=50):
    result = detector.update(x)

    # 変化点検知
    run_length = result["most_likely_run_length"]
    prob = result["run_length_dist"][run_length]

    if run_length == 1 and prob > 0.5:
        print(f"時刻 {t} で変化点を検知")
```

### カスタムモデルの実装

```python
from bocpd.models import PredictiveModel

class MyCustomModel(PredictiveModel):
    def fit_empirical(self, data):
        # 経験ベイズでパラメータを推定
        pass

    def predict(self, x):
        # 予測分布の対数確率を計算
        pass

    def update(self, x):
        # 事後分布を更新（新しいインスタンスを返す）
        return MyCustomModel(updated_params)

    def copy(self):
        return MyCustomModel(self.params)
```

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

## アルゴリズム

BOCPDは、Adams & MacKay (2007) のベイズオンライン変化点検知アルゴリズムを実装しています。

**基本的な考え方:**
1. ランレングス（最後の変化点からの経過時間）の分布を維持
2. 新しいデータが観測されるたびに、各ランレングス仮説の尤度を計算
3. ベイズの定理でランレングス分布を更新
4. ランレングス0（変化点発生）の確率が高い場合、変化点と判定

**数値安定性:**
- 全ての確率計算は対数空間で実行
- log-sum-exp トリックで数値オーバーフロー/アンダーフローを防止

## 開発

### テスト実行

```bash
uv run pytest tests/
```

### コードフォーマット

```bash
uv run black bocpd/
uv run ruff check bocpd/
```

### 型チェック

```bash
uv run mypy bocpd/
```

## 今後の拡張予定

- [ ] 他のベイズ共役モデル（ポアソン-ガンマ、ベルヌーイ-ベータ）
- [ ] 多変量ガウスモデル
- [ ] 可視化ユーティリティの充実
- [ ] パフォーマンス最適化
- [ ] より高度なハザード関数

## 参考文献

Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. arXiv preprint arXiv:0710.3742.

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！バグ報告や機能要望は、GitHubのIssuesでお願いします。
