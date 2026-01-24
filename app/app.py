from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
class MockEncoder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.classes_ = np.array(sorted(mapping, key=mapping.get))
        
    def transform(self, x):
        return x.map(self.mapping)
        
    def inverse_transform(self, x):
        inv_map = {v: k for k, v in self.mapping.items()}
        return np.array([inv_map[i] for i in x])

# 1. モデルとエンコーダーを読み込む
model_data = joblib.load('model_improved.pkl')
model = model_data['model']
feature_cols = model_data['feature_cols']
encoders = joblib.load('encoders_improved.pkl')

@app.route('/')
def index():
    # UIのHTMLを表示する
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # 1. AIくんの辞書に合わせたマッピング
    activity_map = {'低': 'Low', '中': 'Moderate', '高': 'High'}
    weather_map = {'晴': 'Hot', '雨': 'Cold', '曇': 'Normal'}
    
    # 2. データの組み立て
    input_df = pd.DataFrame([{
        'Age': int(data['age']),
        'Weight (kg)': float(data['weight']),
        'Daily Water Intake (liters)': float(data['water']),
        'Physical Activity Level': activity_map.get(data['activity'], data['activity']),
        'Weather': weather_map.get(data['weather'], data['weather'])
    }])

    # --- デバッグ用：AIに渡す直前の「文字」を確認 ---
    print(f"DEBUG - AIに渡すデータ:\n{input_df}")

    # 3. エンコード（数字変換）
    for col in ['Physical Activity Level', 'Weather']:
        input_df[col] = encoders[col].transform(input_df[col])

    # --- デバッグ用：AIに渡す直前の「数字」を確認 ---
    print(f"DEBUG - エンコード後の数字:\n{input_df}")

    # 4. 予測実行
    prediction_idx = model.predict(input_df)[0]

    current_activity = input_df['Physical Activity Level'].iloc[0] # 0=Low, 1=Moderate, 2=High
    current_water = input_df['Daily Water Intake (liters)'].iloc[0]
    current_weather = input_df['Weather'].iloc[0] # 0=Cold(雨), 1=Normal(曇), 2=Hot(晴)
    
    print(f"DEBUG - 補正前チェック: 活動={current_activity}, 天気={current_weather}, 水={current_water}")

    # --- ルール1：活動量が高いのに水が少ない時 ---
    if current_activity >= 1 and current_water < 1.0:
        print("DEBUG - ⚠️ 運動してるのに水が少ない！強制的に『リスクあり』にします")
        prediction_idx = 1 

    # --- ルール2：晴れ（暑い）なのに水が少ない時 ---
    # 活動量が低くても、晴れ(2)なら 1.2L くらい飲まないと危険！というルールを追加
    elif current_weather == 2 and current_water < 1.2:
        print("DEBUG - ⚠️ 晴れてるのに水が少ない！強制的に『リスクあり』にします")
        prediction_idx = 1
    
    # 5. 【ここが超重要】AIが知っている文字に逆変換して判定
    raw_result = encoders['Hydration Level'].inverse_transform([prediction_idx])[0]
    
    print(f"DEBUG - AIの予測(数字): {prediction_idx}")
    print(f"DEBUG - AIの予測(文字): {raw_result}")

    # 6. 判定結果を日本語にする
    # AIが 'Good' と言ったら「十分」、それ以外（Poor）なら「リスクあり」
    if raw_result == 'Good':
        final_result = '水分補給は十分です 🟢'
    else:
        final_result = '脱水のリスクがあります 🔴'

    return jsonify({'result': final_result})


    

if __name__ == '__main__':
    app.run(debug=True)
