"""
train_model.py
水分補給レベル（Hydration Level）を予測する分類モデルを学習・保存するスクリプト
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ファイルパス設定
DATA_PATH = 'Daily_Water_Intake.csv'
MODEL_PATH = 'model.pkl'
ENCODERS_PATH = 'encoders.pkl'

def load_data(filepath):
    """CSVファイルからデータを読み込む"""
    df = pd.read_csv(filepath)
    print(f"データ読み込み完了: {len(df)}行")
    print(f"カラム: {list(df.columns)}")
    return df

def preprocess_data(df):
    """データの前処理を行う"""
    # エンコーダーを保存する辞書
    encoders = {}
    
    # カテゴリカル変数のエンコーディング
    categorical_cols = ['Gender', 'Physical Activity Level', 'Weather']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"{col} エンコード: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # ターゲット変数のエンコーディング
    target_encoder = LabelEncoder()
    df['Hydration Level'] = target_encoder.fit_transform(df['Hydration Level'])
    encoders['Hydration Level'] = target_encoder
    print(f"Hydration Level エンコード: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    
    return df, encoders

def train_model(df):
    """モデルを学習する"""
    # 特徴量とターゲットの分離
    feature_cols = ['Age', 'Gender', 'Weight (kg)', 'Daily Water Intake (liters)', 
                    'Physical Activity Level', 'Weather']
    X = df[feature_cols]
    y = df['Hydration Level']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n学習データ: {len(X_train)}件, テストデータ: {len(X_test)}件")
    
    # モデル学習
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("モデル学習完了")
    
    return model, X_test, y_test, feature_cols

def evaluate_model(model, X_test, y_test, encoders):
    """モデルを評価する"""
    y_pred = model.predict(X_test)
    
    # 評価指標の計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("モデル評価結果")
    print("="*50)
    print(f"正解率 (Accuracy):  {accuracy:.4f}")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"再現率 (Recall):    {recall:.4f}")
    print(f"F1スコア:           {f1:.4f}")
    
    # 分類レポート
    target_names = encoders['Hydration Level'].classes_
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("混同行列:")
    print(cm)
    
    # 混同行列の可視化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title('混同行列 (Confusion Matrix)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\n混同行列を confusion_matrix.png に保存しました")
    
    return accuracy, precision, recall, f1

def plot_feature_importance(model, feature_cols):
    """特徴量の重要度を可視化する"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('特徴量の重要度 (Feature Importance)')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('特徴量')
    plt.ylabel('重要度')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("特徴量の重要度を feature_importance.png に保存しました")

def save_model(model, encoders, feature_cols, model_path, encoders_path):
    """モデルとエンコーダーを保存する"""
    # モデルと特徴量リストを一緒に保存
    model_data = {
        'model': model,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, model_path)
    joblib.dump(encoders, encoders_path)
    print(f"\nモデルを {model_path} に保存しました")
    print(f"エンコーダーを {encoders_path} に保存しました")

def main():
    """メイン処理"""
    print("="*50)
    print("水分補給レベル分類モデル 学習スクリプト")
    print("="*50)
    
    # データ読み込み
    df = load_data(DATA_PATH)
    
    # 前処理
    df, encoders = preprocess_data(df)
    
    # モデル学習
    model, X_test, y_test, feature_cols = train_model(df)
    
    # 評価
    evaluate_model(model, X_test, y_test, encoders)
    
    # 特徴量の重要度
    plot_feature_importance(model, feature_cols)
    
    # モデル保存
    save_model(model, encoders, feature_cols, MODEL_PATH, ENCODERS_PATH)
    
    print("\n" + "="*50)
    print("処理完了")
    print("="*50)

if __name__ == "__main__":
    main()
