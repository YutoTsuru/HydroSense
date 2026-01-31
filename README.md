---
title: HydroSense
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5000
---

# HydroSense - 水分補給診断アプリ

AIを使って、年齢・体重・活動量・天候から水分補給の状態を診断するアプリです。

## 使い方

1. 年齢、体重、1日の水分摂取量を入力
2. 身体活動レベルと天候を選択
3. 「診断する」ボタンをクリック
4. AIが水分補給の状態を判定してアドバイスを表示

## 技術スタック

- **Flask**: Webアプリケーションフレームワーク
- **scikit-learn**: 機械学習モデル
- **pandas/numpy**: データ処理