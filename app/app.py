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
    
    # 1. データの準備
    activity_map = {'低': 'Low', '中': 'Moderate', '高': 'High'}
    weather_map = {'晴': 'Hot', '雨': 'Cold', '曇': 'Normal'}
    
    input_df = pd.DataFrame([{
        'Age': int(data['age']),
        'Weight (kg)': float(data['weight']),
        'Daily Water Intake (liters)': float(data['water']),
        'Physical Activity Level': activity_map.get(data['activity'], data['activity']),
        'Weather': weather_map.get(data['weather'], data['weather'])
    }])

    # 2. 数字に変換
    for col in ['Physical Activity Level', 'Weather']:
        input_df[col] = encoders[col].transform(input_df[col])

    # =================================================================
    # 👑 最強ハイブリッド判定ロジック（現実的調整版） 👑
    # =================================================================
    
    current_activity = input_df['Physical Activity Level'].iloc[0] # 0=Low, 1=Moderate, 2=High
    current_water = input_df['Daily Water Intake (liters)'].iloc[0]
    current_weather = input_df['Weather'].iloc[0] # 0=Cold, 1=Normal, 2=Hot

    # AIに「確率」を聞く
    probs = model.predict_proba(input_df)
    poor_prob = probs[0][1]
    
    print(f"DEBUG - AIリスク確率: {poor_prob * 100:.1f}%")

    prediction_idx = 0 

    # 🛡️ STEP 1: ルール（安全装置）チェック
    
    # 【Lv.MAX】 猛暑(2) かつ 激しい運動(2) -> 1.8L未満ならアウト
    if current_weather == 2 and current_activity == 2 and current_water < 1.8:
        prediction_idx = 1
        print("DEBUG - [判定] ルール：猛暑で激しい運動！1.8L未満なので強制Poor")

    # 【Lv.High】 活動量が高い(2) -> 1.2L未満ならアウト
    elif current_activity == 2 and current_water < 1.2:
        prediction_idx = 1
        print("DEBUG - [判定] ルール：激しい運動のため1.2L未満は強制Poor")

    # 【Lv.Middle】 猛暑(2) -> 1.0L未満ならアウト
    elif current_weather == 2 and current_water < 0.8:
        prediction_idx = 1
        print("DEBUG - [判定] ルール：猛暑のため1.0L未満は強制Poor")
        
    # 【Lv.Low】 救済ゾーン：活動量「低(0)」かつ猛暑じゃない -> 0.5Lあればセーフ
    elif current_activity == 0 and current_weather != 2 and current_water >= 0.5:
        prediction_idx = 0
        print("DEBUG - [判定] ルール：安静時救済。0.5L以上でGood")

    # 🤖 STEP 2: AI閾値チェック
    else:
        THRESHOLD = 0.35
        if poor_prob > THRESHOLD:
            prediction_idx = 1
            print(f"DEBUG - [判定] AI閾値：確率{poor_prob:.2f} > {THRESHOLD} なのでPoor")
        else:
            prediction_idx = 0
            print(f"DEBUG - [判定] AI閾値：確率{poor_prob:.2f} <= {THRESHOLD} なのでGood")

    # =================================================================

    # 3. 結果作成 & アドバイス
    raw_result = encoders['Hydration Level'].inverse_transform([prediction_idx])[0]
    
    advice_message = ""
    if prediction_idx == 0:
        advice_message = "素晴らしい水分管理です！この調子でキープしましょう✨"
    else:
        if current_weather == 2:
            advice_message = "今日は暑いので、喉が渇く前にこまめに水を飲んでください！☀️💦"
        elif current_activity == 2:
            advice_message = "運動で汗をかいています！スポーツドリンクなどで塩分も補給してね🏃‍♂️"
        else:
            advice_message = "水分が不足しています。コップ1杯の水を今すぐ飲みましょう！🚰"

    final_result = '水分補給は十分です 🟢' if prediction_idx == 0 else '脱水のリスクがあります 🔴'

    return jsonify({
        'result': final_result,
        'advice': advice_message
    })

    

if __name__ == '__main__':
    app.run(debug=True)
