# HydroSense

水分補給レベル（Hydration Level）を予測する機械学習プロジェクト

## 概要
日々の活動データと生理学的データに基づいて、水分補給が「良好 (Good)」か「不良 (Poor)」かを分類するRandom Forestモデルを提供します。

## データセット
`Daily_Water_Intake.csv` を使用します。

| カラム名 | 説明 |
|---------|------|
| Age | 年齢 |
| Gender | 性別 (Male/Female) |
| Weight (kg) | 体重 |
| Daily Water Intake (liters) | 1日の水分摂取量 |
| Physical Activity Level | 身体活動レベル (Low/Moderate/High) |
| Weather | 天候 (Cold/Normal/Hot) |
| Hydration Level | 水分補給レベル (Good/Poor) ※ターゲット変数 |

## ファイル構成
```
├── Daily_Water_Intake.csv  # データセット
├── train_model.py          # モデル学習スクリプト
├── inference.py            # 推論モジュール
├── model.pkl               # 学習済みモデル
├── encoders.pkl            # エンコーダー
├── confusion_matrix.png    # 混同行列
└── feature_importance.png  # 特徴量の重要度
```

## 使用方法

### モデル学習
```bash
python train_model.py
```

### 推論（Webアプリ連携用）
```python
from inference import HydrationPredictor

predictor = HydrationPredictor()

result = predictor.predict({
    'Age': 30,
    'Gender': 'Male',
    'Weight (kg)': 70,
    'Daily Water Intake (liters)': 2.5,
    'Physical Activity Level': 'Moderate',
    'Weather': 'Normal'
})
print(result)  # 'Good' or 'Poor'
```

## 依存関係
- pandas
- scikit-learn
- joblib
- matplotlib
- seaborn

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```