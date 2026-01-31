# HydroSense

水分補給レベル（Hydration Level）を予測する機械学習プロジェクト

## 概要
日々の活動データと生理学的データに基づいて、水分補給が「良好 (Good)」か「不良 (Poor)」かを分類するRandom Forestモデルを提供します。

## データセット
`Daily_Water_Intake.csv` を使用します（30,001件のデータ）。

| カラム名 | 説明 | 型 |
|---------|------|-----|
| Age | 年齢 | 数値 (18-69) |
| Gender | 性別 | カテゴリ (Male/Female) |
| Weight (kg) | 体重 | 数値 (45-109) |
| Daily Water Intake (liters) | 1日の水分摂取量 | 数値 (1.5-5.43) |
| Physical Activity Level | 身体活動レベル | カテゴリ (Low/Moderate/High) |
| Weather | 天候 | カテゴリ (Cold/Normal/Hot) |
| Hydration Level | 水分補給レベル | カテゴリ (Good/Poor) ※ターゲット変数 |

## データ前処理

### 1. カテゴリ変数のエンコーディング
Label Encodingを使用してカテゴリ変数を数値に変換：

| 変数 | エンコード |
|-----|----------|
| Gender | Female → 0, Male → 1 |
| Physical Activity Level | High → 0, Low → 1, Moderate → 2 |
| Weather | Cold → 0, Hot → 1, Normal → 2 |
| Hydration Level (ターゲット) | Good → 0, Poor → 1 |

### 2. 数値変数
Random Forestはスケーリングに対して堅牢なため、数値変数（Age, Weight, Daily Water Intake）は正規化せずそのまま使用。

### 3. データ分割
- 学習データ: 80%
- テストデータ: 20%
- 層化抽出（stratify）を使用してクラス分布を維持

## ファイル構成
```
├── Daily_Water_Intake.csv  # データセット
├── train_model.py          # モデル学習スクリプト
├── inference.py            # 推論モジュール
├── model.pkl               # 学習済みモデル
├── encoders.pkl            # エンコーダー（推論時の変換に使用）
├── confusion_matrix.png    # 混同行列
└── feature_importance.png  # 特徴量の重要度
```

## 使用方法

### モデル学習
```bash
python train_model.py
```

### 改善版モデル学習
```bash
pip install imbalanced-learn
python train_model_improved.py
```

改善版では以下の機能が追加されています：
- **SMOTE**: クラス不均衡対策
- **Gender除外**: 低重要度特徴量の削除
- **交差検証**: 5-Fold CVでの過学習チェック
- **GridSearchCV**: ハイパーパラメータ最適化

---

## 改善版モデルの処理結果

### 1. データ読み込みと前処理

30,001件のデータを読み込み、カテゴリ変数を数値にエンコードしました。改善版ではGender特徴量を除外しています。これは元のモデルでGenderの重要度がわずか0.5%であり、予測にほとんど寄与していなかったためです。

### 2. クラス不均衡対策（SMOTE適用）

元のデータセットではGood（水分補給良好）とPoor（水分補給不良）の比率が約4:1と不均衡でした。SMOTEを適用することで、少数派クラス（Poor）を合成サンプルで増加させ、学習データのバランスを改善しました。

- **SMOTE適用前**: Good 19,185件 / Poor 4,816件
- **SMOTE適用後**: Good 19,185件 / Poor 19,185件（均等化）

### 3. ハイパーパラメータチューニング（GridSearchCV）

GridSearchCVを使用して、Random Forestの最適なハイパーパラメータを3分割交差検証で探索しました。探索したパラメータは以下の通りです：

- `n_estimators`: 50, 100, 200
- `max_depth`: 5, 10, 15, None
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4

計108通りの組み合わせを評価した結果、最適パラメータは以下の通りでした：

| パラメータ | 最適値 |
|-----------|--------|
| n_estimators | 200 |
| max_depth | 15 |
| min_samples_split | 2 |
| min_samples_leaf | 1 |

### 4. 交差検証結果

5分割交差検証（Stratified K-Fold）を実施し、モデルの汎化性能を確認しました。

| Fold | 精度 |
|------|------|
| Fold 1 | 99.12% |
| Fold 2 | 99.35% |
| Fold 3 | 99.35% |
| Fold 4 | 99.32% |
| Fold 5 | 99.32% |
| **平均** | **99.29% (±0.18%)** |

全てのFoldで99%以上の精度を達成しており、標準偏差も0.09%と非常に小さいことから、過学習の兆候はなく、モデルは安定して汎化できていることが確認されました。

### 5. テストセットでの評価結果

テストデータ（6,001件、全体の20%）での評価結果は以下の通りです：

| 評価指標 | 値 |
|---------|------|
| 正解率（Accuracy） | 99.08% |
| 適合率（Precision） | 99.08% |
| 再現率（Recall） | 99.08% |
| F1スコア | 99.08% |

#### クラス別分類レポート

| クラス | Precision | Recall | F1-Score | サンプル数 |
|--------|-----------|--------|----------|------------|
| Good | 1.00 | 0.99 | 0.99 | 4,783 |
| Poor | 0.97 | 0.98 | 0.98 | 1,217 |

### 6. 混同行列

|  | Good (予測) | Poor (予測) |
|---|---|---|
| **Good (実際)** | 4,748 | 35 |
| **Poor (実際)** | 20 | 1,197 |

- **Good → Poor の誤分類**: 35件（Goodの0.7%）
- **Poor → Good の誤分類**: 20件（Poorの1.6%）

### 7. 改善前後の比較

| 指標 | 改善前 | 改善後 | 変化 |
|------|--------|--------|------|
| 正解率 | 98.4% | 99.1% | +0.7% |
| Poor→Good 誤分類 | 59件 | 20件 | **66%削減** |
| Good→Poor 誤分類 | 36件 | 35件 | ほぼ同等 |

特に、少数派クラス（Poor）の誤分類が大幅に減少しています。これはSMOTEによるクラス不均衡対策が効果的に機能した結果です。

### 8. 特徴量重要度（Gender除外後）

| 順位 | 特徴量 | 重要度 |
|------|--------|--------|
| 1位 | Daily Water Intake (liters) | 37.1% |
| 2位 | Weight (kg) | 26.0% |
| 3位 | Physical Activity Level | 21.8% |
| 4位 | Weather | 12.0% |
| 5位 | Age | 3.1% |

1日の水分摂取量が最も重要な特徴量であり、体重と身体活動レベルがそれに続きます。天候も一定の影響がありますが、年齢の影響は比較的小さいことがわかります。

---

## Webアプリ連携ガイド

### 必要ファイル
Webアプリ側に以下のファイルを配置してください：
- `inference.py` - 推論モジュール
- `model.pkl` - 学習済みモデル
- `encoders.pkl` - エンコーダー

### 環境構築
```bash
pip install pandas scikit-learn joblib
```

### 基本的な使い方

```python
from inference import HydrationPredictor

# 予測器を初期化（モデルは自動で読み込まれる）
predictor = HydrationPredictor()

# 予測を実行
result = predictor.predict({
    'Age': 30,
    'Gender': 'Male',
    'Weight (kg)': 70,
    'Daily Water Intake (liters)': 2.5,
    'Physical Activity Level': 'Moderate',
    'Weather': 'Normal'
})
print(result)  # 'Good' or 'Poor'

# 予測確率も取得可能
proba = predictor.predict_proba({...})
# {'Good': 0.85, 'Poor': 0.15}
```

### Flask での実装例

```python
from flask import Flask, request, jsonify
from inference import HydrationPredictor

app = Flask(__name__)
predictor = HydrationPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # 入力データの形式
    # {
    #     "Age": 30,
    #     "Gender": "Male",
    #     "Weight (kg)": 70,
    #     "Daily Water Intake (liters)": 2.5,
    #     "Physical Activity Level": "Moderate",
    #     "Weather": "Normal"
    # }
    
    result = predictor.predict(data)
    proba = predictor.predict_proba(data)
    
    return jsonify({
        'prediction': result,
        'probability': proba
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI での実装例

```python
from fastapi import FastAPI
from pydantic import BaseModel
from inference import HydrationPredictor

app = FastAPI()
predictor = HydrationPredictor()

class PredictionInput(BaseModel):
    Age: int
    Gender: str  # 'Male' or 'Female'
    Weight_kg: float  # JSONでは 'Weight (kg)' として受け取る
    Daily_Water_Intake_liters: float
    Physical_Activity_Level: str  # 'Low', 'Moderate', 'High'
    Weather: str  # 'Cold', 'Normal', 'Hot'

@app.post('/predict')
def predict(input_data: PredictionInput):
    # 入力をinference.pyが期待する形式に変換
    data = {
        'Age': input_data.Age,
        'Gender': input_data.Gender,
        'Weight (kg)': input_data.Weight_kg,
        'Daily Water Intake (liters)': input_data.Daily_Water_Intake_liters,
        'Physical Activity Level': input_data.Physical_Activity_Level,
        'Weather': input_data.Weather
    }
    
    result = predictor.predict(data)
    proba = predictor.predict_proba(data)
    
    return {
        'prediction': result,
        'probability': proba
    }
```

### 入力値の制約

| パラメータ | 型 | 許容値 |
|-----------|-----|--------|
| Age | int | 18-69（推奨） |
| Gender | str | `'Male'` または `'Female'` |
| Weight (kg) | float | 45-109（推奨） |
| Daily Water Intake (liters) | float | 1.5-5.5（推奨） |
| Physical Activity Level | str | `'Low'`, `'Moderate'`, `'High'` |
| Weather | str | `'Cold'`, `'Normal'`, `'Hot'` |

### エラーハンドリング
`inference.py` は未知のカテゴリ値に対して `ValueError` を発生させます。
Webアプリ側で適切にエラーハンドリングしてください。

```python
try:
    result = predictor.predict(data)
except ValueError as e:
    return jsonify({'error': str(e)}), 400
except FileNotFoundError as e:
    return jsonify({'error': 'Model files not found'}), 500
```

## 依存関係
```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```