"""
경기기후 플랫폼 - LightGBM 모델 학습 스크립트

수집된 데이터(climate_learning_data.json)를 이용하여
복원력 점수를 예측하는 LightGBM 모델을 학습합니다.

학습 후 SHAP 설명 가능성을 통해 각 지표의 기여도를 분석합니다.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import joblib

# ============================================
# 데이터 로드
# ============================================

def load_training_data(log_file="climate_learning_data.json"):
    """
    climate_learning_data.json에서 학습 데이터 로드

    반환 형식:
    - X: (N, 8) 지표 배열
    - y: (N,) 복원력 점수 배열
    """
    print("=" * 60)
    print("📂 데이터 로드 중...")
    print("=" * 60)

    with open(log_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    # 예측 로그만 필터링
    prediction_logs = [log for log in logs if log.get('endpoint') == '/api/predict']

    if len(prediction_logs) == 0:
        raise ValueError("❌ 학습 데이터가 없습니다. 앱을 사용하여 데이터를 수집하세요.")

    print(f"✅ 총 {len(prediction_logs)}개 레코드 로드됨")

    # Feature 배열 생성 (8개 지표)
    X = []
    y = []

    for log in prediction_logs:
        features = [
            log['features']['heatDays'],
            log['features']['floodFrequency'],
            log['features']['imperviousRate'],
            log['features']['greenRate'],
            log['features']['heatIsland'],
            log['features']['elderlyRatio'],
            log['features']['shelterAccess'],
            log['features']['medicalAccess']
        ]
        X.append(features)
        y.append(log['prediction'])  # 실제로는 사용자 피드백 점수로 교체

    X = np.array(X)
    y = np.array(y)

    print(f"📊 Feature shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
    print()

    return X, y

# ============================================
# 모델 학습
# ============================================

def train_lightgbm_model(X, y, test_size=0.2, random_state=42):
    """
    LightGBM 모델 학습

    하이퍼파라미터:
    - objective: regression (복원력 점수 예측)
    - metric: rmse, mae
    - num_leaves: 31 (트리 복잡도)
    - learning_rate: 0.05
    - n_estimators: 100
    """
    print("=" * 60)
    print("🎓 LightGBM 모델 학습 시작")
    print("=" * 60)

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print()

    # LightGBM 파라미터
    params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'random_state': random_state,
        'verbose': -1
    }

    print("🔧 하이퍼파라미터:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    print()

    # 모델 학습
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse'
    )

    # 예측 및 평가
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=" * 60)
    print("📈 모델 평가 결과")
    print("=" * 60)
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print()

    return model, X_test, y_test

# ============================================
# SHAP 설명
# ============================================

def explain_with_shap(model, X_test, feature_names):
    """
    SHAP을 이용한 모델 설명

    각 지표의 복원력 점수에 대한 기여도를 계산합니다.
    """
    print("=" * 60)
    print("🔍 SHAP 설명 생성 중...")
    print("=" * 60)

    # SHAP Explainer 생성
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 평균 SHAP 기여도
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    print("📊 지표별 평균 SHAP 기여도 (절댓값):")
    print()
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name:20s}: {mean_abs_shap[i]:.4f}")
    print()

    # 중요도 순위
    importance_ranking = np.argsort(-mean_abs_shap)
    print("🏆 중요도 순위:")
    print()
    for rank, idx in enumerate(importance_ranking):
        print(f"  #{rank+1}. {feature_names[idx]:20s} (기여도: {mean_abs_shap[idx]:.4f})")
    print()

    return explainer, shap_values

# ============================================
# 모델 저장
# ============================================

def save_model(model, explainer, model_path="climate_model.pkl", explainer_path="shap_explainer.pkl"):
    """모델 및 SHAP Explainer 저장"""
    print("=" * 60)
    print("💾 모델 저장 중...")
    print("=" * 60)

    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)

    print(f"✅ 모델 저장 완료: {model_path}")
    print(f"✅ SHAP Explainer 저장 완료: {explainer_path}")
    print()

# ============================================
# 메인 실행
# ============================================

def main():
    """메인 학습 파이프라인"""
    print()
    print("=" * 60)
    print("🚀 경기기후 플랫폼 - LightGBM 모델 학습")
    print("=" * 60)
    print(f"⏰ 시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # 지표명
    feature_names = [
        "폭염일수",
        "침수빈도",
        "불투수율",
        "녹지율",
        "열섬지수",
        "고령자비율",
        "쉼터접근성",
        "의료접근성"
    ]

    try:
        # 1. 데이터 로드
        X, y = load_training_data()

        # 최소 데이터 개수 확인
        if X.shape[0] < 30:
            print(f"⚠️ 경고: 데이터가 {X.shape[0]}개로 부족합니다 (권장: 30개 이상)")
            print("   앱을 더 사용하여 데이터를 수집하세요.")
            return

        # 2. 모델 학습
        model, X_test, y_test = train_lightgbm_model(X, y)

        # 3. SHAP 설명
        explainer, shap_values = explain_with_shap(model, X_test, feature_names)

        # 4. 모델 저장
        save_model(model, explainer)

        print("=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print()
        print("다음 단계:")
        print("  1. app.py에서 저장된 모델(climate_model.pkl)을 로드하세요")
        print("  2. 더미 예측 대신 실제 모델 예측을 사용하도록 수정하세요")
        print("  3. Flask 서버를 재시작하여 ML 예측을 활성화하세요")
        print()

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print("=" * 60)
        print(f"⏰ 종료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()

if __name__ == '__main__':
    main()
