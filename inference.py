"""
inference.py
学習済みモデルを使用して水分補給レベルを予測するモジュール
Webアプリ等からインポートして使用可能
"""

import joblib
import pandas as pd
import numpy as np
import os

# デフォルトのモデルパス
DEFAULT_MODEL_PATH = 'model.pkl'
DEFAULT_ENCODERS_PATH = 'encoders.pkl'


class HydrationPredictor:
    """水分補給レベル予測クラス"""
    
    def __init__(self, model_path=None, encoders_path=None):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス（Noneの場合はデフォルトパスを使用）
            encoders_path: エンコーダーファイルのパス（Noneの場合はデフォルトパスを使用）
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.encoders_path = encoders_path or DEFAULT_ENCODERS_PATH
        self.model = None
        self.encoders = None
        self.feature_cols = None
        
    def load_model(self):
        """保存されたモデルとエンコーダーを読み込む"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
        if not os.path.exists(self.encoders_path):
            raise FileNotFoundError(f"エンコーダーファイルが見つかりません: {self.encoders_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.encoders = joblib.load(self.encoders_path)
        print("モデルとエンコーダーを読み込みました")
        
    def preprocess_input(self, data):
        """
        入力データを前処理する
        
        Args:
            data: 辞書型の入力データ
                {
                    'Age': int,
                    'Gender': str ('Male' or 'Female'),
                    'Weight (kg)': float,
                    'Daily Water Intake (liters)': float,
                    'Physical Activity Level': str ('Low', 'Moderate', or 'High'),
                    'Weather': str ('Cold', 'Normal', or 'Hot')
                }
        
        Returns:
            前処理済みのDataFrame
        """
        # 辞書をDataFrameに変換
        df = pd.DataFrame([data])
        
        # カテゴリカル変数のエンコーディング
        categorical_cols = ['Gender', 'Physical Activity Level', 'Weather']
        for col in categorical_cols:
            if col in df.columns and col in self.encoders:
                encoder = self.encoders[col]
                # 未知のカテゴリに対応
                if df[col].iloc[0] in encoder.classes_:
                    df[col] = encoder.transform(df[col])
                else:
                    raise ValueError(f"未知の値です: {col} = {df[col].iloc[0]}")
        
        # 特徴量の順序を揃える
        df = df[self.feature_cols]
        
        return df
    
    def predict(self, data):
        """
        予測を行う
        
        Args:
            data: 辞書型の入力データ
        
        Returns:
            予測結果 ('Good' or 'Poor')
        """
        if self.model is None:
            self.load_model()
        
        # 前処理
        X = self.preprocess_input(data)
        
        # 予測
        prediction = self.model.predict(X)[0]
        
        # ラベルに変換
        target_encoder = self.encoders['Hydration Level']
        result = target_encoder.inverse_transform([prediction])[0]
        
        return result
    
    def predict_proba(self, data):
        """
        予測確率を返す
        
        Args:
            data: 辞書型の入力データ
        
        Returns:
            各クラスの予測確率の辞書
        """
        if self.model is None:
            self.load_model()
        
        # 前処理
        X = self.preprocess_input(data)
        
        # 予測確率
        proba = self.model.predict_proba(X)[0]
        
        # ラベルと確率を対応付け
        target_encoder = self.encoders['Hydration Level']
        result = {
            label: float(prob) 
            for label, prob in zip(target_encoder.classes_, proba)
        }
        
        return result


def main():
    """テスト用のメイン処理"""
    print("="*50)
    print("推論テスト")
    print("="*50)
    
    # 予測器の初期化
    predictor = HydrationPredictor()
    
    # テストデータ
    test_cases = [
        {
            'Age': 30,
            'Gender': 'Male',
            'Weight (kg)': 70,
            'Daily Water Intake (liters)': 2.5,
            'Physical Activity Level': 'Moderate',
            'Weather': 'Normal'
        },
        {
            'Age': 25,
            'Gender': 'Female',
            'Weight (kg)': 55,
            'Daily Water Intake (liters)': 1.2,
            'Physical Activity Level': 'Low',
            'Weather': 'Hot'
        },
        {
            'Age': 45,
            'Gender': 'Male',
            'Weight (kg)': 85,
            'Daily Water Intake (liters)': 4.0,
            'Physical Activity Level': 'High',
            'Weather': 'Hot'
        }
    ]
    
    for i, data in enumerate(test_cases, 1):
        print(f"\nテストケース {i}:")
        print(f"  入力: {data}")
        
        result = predictor.predict(data)
        proba = predictor.predict_proba(data)
        
        print(f"  予測結果: {result}")
        print(f"  予測確率: {proba}")
    
    print("\n" + "="*50)
    print("テスト完了")
    print("="*50)


if __name__ == "__main__":
    main()
