"""
train_model_improved.py
改善版: 水分補給レベル分類モデルの学習スクリプト

改善点:
1. SMOTEによるクラス不均衡対策
2. Gender特徴量の除外オプション
3. 交差検証による過学習チェック
4. GridSearchCVによるハイパーパラメータチューニング
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ファイルパス設定
DATA_PATH = 'Daily_Water_Intake.csv'
MODEL_PATH = 'model_improved.pkl'
ENCODERS_PATH = 'encoders_improved.pkl'

# 改善オプション
EXCLUDE_GENDER = True  # Genderを除外するか
USE_SMOTE = True       # SMOTEを使用するか
USE_GRIDSEARCH = True  # GridSearchを使用するか

def load_data(filepath):
    """CSVファイルからデータを読み込む"""
    df = pd.read_csv(filepath)
    print(f"データ読み込み完了: {len(df)}行")
    return df

def preprocess_data(df, exclude_gender=False):
    """データの前処理を行う"""
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

def apply_smote(X_train, y_train):
    """SMOTEでオーバーサンプリング"""
    print("\n--- SMOTE適用 ---")
    print(f"適用前: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"適用後: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    return X_resampled, y_resampled

def cross_validate_model(model, X, y, cv=5):
    """交差検証を実行"""
    print("\n--- 交差検証 (5-Fold) ---")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"各Fold精度: {scores}")
    print(f"平均精度: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # 可視化
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, cv + 1), scores, color='steelblue', edgecolor='black')
    plt.axhline(y=scores.mean(), color='red', linestyle='--', label=f'平均: {scores.mean():.4f}')
    plt.xlabel('Fold')
    plt.ylabel('精度')
    plt.title('交差検証結果 (5-Fold Cross Validation)')
    plt.legend()
    plt.ylim(0.9, 1.0)
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=150)
    print("交差検証結果を cv_results.png に保存しました")
    
    return scores

def grid_search_tuning(X_train, y_train):
    """GridSearchでハイパーパラメータチューニング"""
    print("\n--- GridSearchCV ハイパーパラメータチューニング ---")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\n最適パラメータ: {grid_search.best_params_}")
    print(f"最高スコア: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_model(df, exclude_gender=False, use_smote=True, use_gridsearch=True):
    """モデルを学習する"""
    # 特徴量の選択
    if exclude_gender:
        feature_cols = ['Age', 'Weight (kg)', 'Daily Water Intake (liters)', 
                        'Physical Activity Level', 'Weather']
        print("\n※ Gender を特徴量から除外しました")
    else:
        feature_cols = ['Age', 'Gender', 'Weight (kg)', 'Daily Water Intake (liters)', 
                        'Physical Activity Level', 'Weather']
    
    X = df[feature_cols]
    y = df['Hydration Level']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n学習データ: {len(X_train)}件, テストデータ: {len(X_test)}件")
    
    # SMOTE適用
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # GridSearch または 標準モデル
    if use_gridsearch:
        model, best_params = grid_search_tuning(X_train, y_train)
    else:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_params = None
    
    print("モデル学習完了")
    
    return model, X_train, X_test, y_train, y_test, feature_cols, best_params

def evaluate_model(model, X_test, y_test, encoders):
    """モデルを評価する"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("モデル評価結果 (改善版)")
    print("="*50)
    print(f"正解率 (Accuracy):  {accuracy:.4f}")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"再現率 (Recall):    {recall:.4f}")
    print(f"F1スコア:           {f1:.4f}")
    
    target_names = encoders['Hydration Level'].classes_
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("混同行列:")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title('混同行列 (改善版モデル)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=150)
    print("\n混同行列を confusion_matrix_improved.png に保存しました")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def plot_feature_importance(model, feature_cols):
    """特徴量の重要度を可視化する"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('特徴量の重要度 (改善版モデル)')
    plt.bar(range(len(importance)), importance[indices], color='steelblue')
    plt.xticks(range(len(importance)), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('特徴量')
    plt.ylabel('重要度')
    plt.tight_layout()
    plt.savefig('feature_importance_improved.png', dpi=150)
    print("特徴量の重要度を feature_importance_improved.png に保存しました")

def save_model(model, encoders, feature_cols, best_params, model_path, encoders_path):
    """モデルとエンコーダーを保存する"""
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'best_params': best_params
    }
    joblib.dump(model_data, model_path)
    joblib.dump(encoders, encoders_path)
    print(f"\nモデルを {model_path} に保存しました")
    print(f"エンコーダーを {encoders_path} に保存しました")

def compare_with_original():
    """元のモデルとの比較"""
    print("\n" + "="*50)
    print("改善前後の比較")
    print("="*50)
    print("※ 元のモデル精度: 約98.4%")
    print("改善版では交差検証でロバスト性を確認し、")
    print("SMOTEでクラス不均衡に対応しています。")

def main():
    """メイン処理"""
    print("="*60)
    print("水分補給レベル分類モデル 改善版学習スクリプト")
    print("="*60)
    print(f"\n設定:")
    print(f"  - Gender除外: {EXCLUDE_GENDER}")
    print(f"  - SMOTE使用: {USE_SMOTE}")
    print(f"  - GridSearch使用: {USE_GRIDSEARCH}")
    
    # データ読み込み
    df = load_data(DATA_PATH)
    
    # 前処理
    df, encoders = preprocess_data(df, exclude_gender=EXCLUDE_GENDER)
    
    # モデル学習
    model, X_train, X_test, y_train, y_test, feature_cols, best_params = train_model(
        df, 
        exclude_gender=EXCLUDE_GENDER,
        use_smote=USE_SMOTE,
        use_gridsearch=USE_GRIDSEARCH
    )
    
    # 交差検証
    X_all = df[feature_cols]
    y_all = df['Hydration Level']
    cross_validate_model(model, X_all, y_all)
    
    # 評価
    metrics = evaluate_model(model, X_test, y_test, encoders)
    
    # 特徴量の重要度
    plot_feature_importance(model, feature_cols)
    
    # モデル保存
    save_model(model, encoders, feature_cols, best_params, MODEL_PATH, ENCODERS_PATH)
    
    # 比較
    compare_with_original()
    
    print("\n" + "="*60)
    print("処理完了")
    print("="*60)

if __name__ == "__main__":
    main()
