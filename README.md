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