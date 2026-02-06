# 경기기후 플랫폼 - ML API 서버

LightGBM 예측, SHAP 설명, 데이터 로깅을 위한 Flask 서버입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# ml-server 디렉토리로 이동
cd ml-server

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 서버 시작

```bash
# Flask 서버 시작 (포트 5000)
python app.py
```

서버가 성공적으로 시작되면 다음과 같은 메시지가 표시됩니다:

```
============================================================
🚀 경기기후 플랫폼 ML API 서버 시작
============================================================
📊 데이터 로그 파일: climate_learning_data.json
🌐 서버 주소: http://localhost:5000
🔗 Health Check: http://localhost:5000/health
📈 데이터 통계: http://localhost:5000/api/logs/stats
============================================================
```

### 3. 서버 상태 확인

브라우저에서 http://localhost:5000/health 를 열어 서버 상태를 확인하세요.

```json
{
  "status": "healthy",
  "model_status": "dummy",
  "timestamp": "2026-01-16T..."
}
```

## 📊 데이터 수집 과정

### 자동 수집

프론트엔드 앱을 사용하면 자동으로 데이터가 수집됩니다:

1. 사용자가 지도에서 지역을 클릭
2. `UnifiedActionBoard.tsx`의 `mlActive` 상태가 `true`로 변경
3. `mlPipelineIntegrated.ts`의 `fetchSHAPValues` 및 `predictWithLightGBM` 호출
4. ML 서버(`app.py`)가 데이터를 수신하고 `climate_learning_data.json`에 저장

### 데이터 형식

`climate_learning_data.json`에 저장되는 데이터 형식:

```json
[
  {
    "timestamp": "2026-01-16T10:30:00.000Z",
    "endpoint": "/api/predict",
    "features": {
      "heatDays": 25,
      "floodFrequency": 3,
      "imperviousRate": 60,
      "greenRate": 25,
      "heatIsland": 3.5,
      "elderlyRatio": 18,
      "shelterAccess": 450,
      "medicalAccess": 70
    },
    "prediction": 62.5
  },
  ...
]
```

## 📈 데이터 통계 확인

수집된 데이터 통계 확인:

```bash
# 브라우저에서 열기
http://localhost:5000/api/logs/stats

# 또는 curl로 확인
curl http://localhost:5000/api/logs/stats
```

응답 예시:

```json
{
  "total_records": 150,
  "prediction_records": 120,
  "avg_prediction": 65.3,
  "features_avg": {
    "heatDays": 23.5,
    "floodFrequency": 2.8,
    "imperviousRate": 55.2,
    ...
  },
  "first_record": "2026-01-15T09:00:00.000Z",
  "last_record": "2026-01-16T15:30:00.000Z",
  "ready_for_training": true
}
```

## 🎓 모델 학습

### 데이터 수집 완료 후 (30개 이상 권장)

```bash
# 학습 스크립트 실행
python train_model.py
```

학습 과정:

1. `climate_learning_data.json`에서 데이터 로드
2. LightGBM 회귀 모델 학습 (Train/Test 분할)
3. SHAP 기여도 분석
4. 모델 및 Explainer 저장 (`climate_model.pkl`, `shap_explainer.pkl`)

### 학습 결과 예시

```
============================================================
🎓 LightGBM 모델 학습 시작
============================================================
📊 Train set: 96 samples
📊 Test set: 24 samples

============================================================
📈 모델 평가 결과
============================================================
  RMSE: 3.45
  MAE: 2.67
  R² Score: 0.8732

============================================================
🔍 SHAP 설명 생성 중...
============================================================
📊 지표별 평균 SHAP 기여도 (절댓값):

  1. 폭염일수              : 0.2341
  2. 침수빈도              : 0.1876
  3. 불투수율              : 0.1654
  ...

🏆 중요도 순위:

  #1. 폭염일수              (기여도: 0.2341)
  #2. 침수빈도              (기여도: 0.1876)
  #3. 불투수율              (기여도: 0.1654)
```

## 🔄 실제 모델로 전환

학습 완료 후 `app.py`를 수정하여 실제 모델을 사용하도록 변경:

```python
# app.py 상단에 추가
import joblib

# 모델 로드
model = joblib.load("climate_model.pkl")
explainer = joblib.load("shap_explainer.pkl")

# predict() 함수에서 더미 예측 대신 실제 예측 사용
def predict():
    # ...
    # prediction = calculate_dummy_prediction(features)  # 제거
    prediction = model.predict([features])[0]  # 실제 모델 사용
    # ...
```

## 🎯 UI에서 AI 배지 확인

서버가 응답을 주면 프론트엔드에서 자동으로:

1. `mlActive` 상태가 `true`로 변경
2. **🤖 AI 핵심 리스크** 배지 표시 (SHAP 1순위 지표)
3. **#지역_베스트** 태그 표시 (MAB 1순위 솔루션)
4. **AI 정밀 진단 결과** 가이드 문구 표시

## 📡 API 엔드포인트

### POST /api/predict

복원력 점수 예측

**요청:**
```json
{
  "features": [25, 3, 60, 25, 3.5, 18, 450, 70]
}
```

**응답:**
```json
{
  "prediction": 62.5,
  "confidence": 0.75,
  "model_version": "v0.1-dummy",
  "note": "실제 모델 학습 전 더미 예측"
}
```

### POST /api/shap

SHAP 설명 가능성

**요청:**
```json
{
  "features": [25, 3, 60, 25, 18, 450, 70]
}
```

**응답:**
```json
{
  "shap_values": [-12.5, -8.3, -15.2, 10.5, -7.8, 6.2, 9.1],
  "feature_names": ["폭염일수", "침수빈도", ...],
  "note": "실제 모델 학습 전 더미 SHAP"
}
```

### GET /health

서버 상태 확인

### GET /api/logs/stats

수집된 데이터 통계

### GET /api/logs/export

학습용 데이터 내보내기

## 🔧 환경 변수

프론트엔드 `.env` 파일에서 ML 서버 URL 설정:

```env
VITE_LIGHTGBM_API_URL=http://localhost:5000
```

## 🐛 트러블슈팅

### ERR_CONNECTION_REFUSED

**증상:** 프론트엔드 콘솔에 `ERR_CONNECTION_REFUSED` 에러 발생

**원인:** ML 서버가 실행되지 않음

**해결:**
```bash
python app.py
```

### mlActive가 false로 유지됨

**원인:** ML 서버가 3초 내에 응답하지 못함

**확인:**
```bash
curl http://localhost:5000/health
```

### 데이터가 수집되지 않음

**확인:**
1. `climate_learning_data.json` 파일 존재 여부
2. Flask 서버 로그에서 `[LOG] 데이터 저장 완료` 메시지 확인

## 📚 다음 단계

1. ✅ 사용자가 앱을 사용하며 데이터 수집 (30개 이상)
2. ✅ `python train_model.py` 실행하여 모델 학습
3. ✅ `app.py`에서 실제 모델 로드하도록 수정
4. ✅ Flask 서버 재시작
5. ✅ 프론트엔드에서 AI 배지가 정확하게 표시되는지 확인
6. ✅ Supabase `climate_ai_logs` 테이블에 통계 vs ML 점수 비교 데이터 확인

## 💡 팁

- **초기 테스트:** 더미 예측으로도 UI 배지가 정상 작동하는지 먼저 확인
- **데이터 품질:** 사용자 피드백을 받아 `y` 값을 실제 만족도로 교체하면 모델 품질 향상
- **주기적 재학습:** 새로운 데이터가 쌓이면 주기적으로 재학습 실행
