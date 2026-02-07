"""
경기기후 플랫폼 - ML API 서버 (v44_full_ml)
LightGBM 예측, TreeSHAP 설명, DIY 효과 예측을 위한 Flask 서버

🔥 v44_full_ml: 완전한 ML 파이프라인
1. 기후 안전점수: Rule-based + LightGBM 하이브리드
2. SHAP: 실제 LightGBM TreeSHAP (비선형 관계, 상호작용 포착)
3. DIY 효과 예측: 별도 모델로 점수 변화 예측

프론트엔드 규격:
- 8개 지표 입력: heatDays, floodFrequency, imperviousRate, greenRate,
  heatIsland, elderlyRatio, shelterAccess, medicalAccess
- CORS 허용 (개발 환경)
- JSON 로깅 (climate_learning_data.json)
"""

import sys
import io
import os

# Windows cp949 인코딩 문제 해결: PYTHONIOENCODING 환경변수 + stdout/stderr UTF-8 강제
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    # Python 3.6 이하 fallback
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import numpy as np
import threading
import pickle

# ML 라이브러리
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# CORS 설정 - 모든 origin 허용 (개발 환경)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 파일 경로
LOG_FILE = "climate_learning_data.json"
MODEL_FILE = "lightgbm_model.pkl"
SHAP_EXPLAINER_FILE = "shap_explainer.pkl"
DIY_EFFECT_MODEL_FILE = "diy_effect_model.pkl"

# 전역 모델 변수
model = None
shap_explainer = None
diy_effect_model = None  # 🔥 DIY 효과 예측 모델
model_trained = False
diy_model_trained = False

log_lock = threading.Lock()

# ============================================
# 🔥 climate_logic.py 동기화: 3축 가중치
# ============================================
AXIS_WEIGHTS_CLIMATE = {
    "time": 0.30,      # 시간축: 노출/빈도 (30%)
    "space": 0.40,     # 공간축: 물리적 민감도 (40%)
    "context": 0.30,   # 맥락축: 적응 능력 (30%)
}

# ============================================
# 🔥 climate_logic.py 동기화: DIY 정의 (6종)
# effects: 축별 효과 계수 (time은 음수=위험 감소, space/context는 양수=복원력 증가)
# quantitative_effects: 정량적 효과
# evidence: 선행연구 기반 효과 불확실성 및 근거 수준
# ============================================
DIY_ITEMS = {
    # 🔥 v51_realistic: 선행연구 기반 현실적 효과 하향 조정
    # 참고: EPA Cool Roof Calculator, IPCC AR6, 국토부 도시열섬 가이드라인
    # 개별 DIY 1개 = 약 0.5-1.5점 효과 (기존 8-12점에서 대폭 하향)
    # 문제 매칭 보너스 적용 시 최대 2-3점까지
    'cool_roof': {
        'name': '쿨루프',
        'name_en': 'Cool Roof',
        'icon': '🏠',
        'description': '고반사 페인트로 지붕 온도 저감',
        'affects': ['heatDays', 'heatIsland'],
        'base_effect': 0.8,  # v51: 2.5 → 0.8 (선행연구: 쿨루프 1.2°C 저감)
        'effects': {'time': -0.08, 'space': 0.15, 'context': 0.05},
        'quantitative_effects': {
            'temperature_reduction': 10,  # °C (지붕 표면)
            'carbon_absorption': 0,       # kgCO2/년
            'runoff_reduction': 0,        # L/년
            'energy_saving': 450,         # kWh/년
        },
        'cost': 80,           # 만원
        'install_area': 50,   # ㎡
        'evidence': {
            'level': 'high',
            'confidence': 0.85,
            'effect_range': {'min': 0.8, 'max': 1.2},
            'source': 'EPA Cool Roof Calculator 2023; 국토교통부 도시열섬 저감 가이드라인',
            'study_count': 12,
        },
    },
    'rain_planter': {
        'name': '빗물화분',
        'name_en': 'Rain Planter',
        'icon': '🪴',
        'description': '빗물 저류 및 녹지 조성',
        'affects': ['floodFrequency', 'imperviousRate'],
        'base_effect': 0.6,  # v51: 2.0 → 0.6 (개인 단위 효과 미미)
        'effects': {'time': -0.05, 'space': 0.20, 'context': 0.10},
        'quantitative_effects': {
            'temperature_reduction': 1,
            'carbon_absorption': 2.5,
            'runoff_reduction': 80,
            'energy_saving': 0,
        },
        'cost': 30,
        'install_area': 1,
        'evidence': {
            'level': 'medium',
            'confidence': 0.75,
            'effect_range': {'min': 0.7, 'max': 1.3},
            'source': '환경부 LID 기술 가이드라인 2022; 서울시 빗물관리시설 효과분석',
            'study_count': 8,
        },
    },
    'green_curtain': {
        'name': '녹색커튼',
        'name_en': 'Green Curtain',
        'icon': '🌿',
        'description': '덩굴식물로 벽면 녹화',
        'affects': ['greenRate', 'heatIsland'],
        'base_effect': 0.7,  # v51: 1.8 → 0.7 (벽면 녹화 효과)
        'effects': {'time': -0.10, 'space': 0.12, 'context': 0.08},
        'quantitative_effects': {
            'temperature_reduction': 10,
            'carbon_absorption': 5,
            'runoff_reduction': 10,
            'energy_saving': 300,
        },
        'cost': 50,
        'install_area': 10,
        'evidence': {
            'level': 'high',
            'confidence': 0.80,
            'effect_range': {'min': 0.75, 'max': 1.25},
            'source': 'Urban Heat Mitigation PMC 2024; 일본 그린커튼 효과 연구',
            'study_count': 15,
        },
    },
    'permeable_block': {
        'name': '투수블록',
        'name_en': 'Permeable Block',
        'icon': '🧱',
        'description': '빗물 침투형 보도블록',
        'affects': ['imperviousRate', 'floodFrequency'],
        'base_effect': 0.7,  # v51: 2.2 → 0.7 (20㎡ 단위 효과)
        'effects': {'time': -0.08, 'space': 0.25, 'context': 0.03},
        'quantitative_effects': {
            'temperature_reduction': 3,
            'carbon_absorption': 0,
            'runoff_reduction': 70,
            'energy_saving': 0,
        },
        'cost': 100,
        'install_area': 20,
        'evidence': {
            'level': 'high',
            'confidence': 0.90,
            'effect_range': {'min': 0.85, 'max': 1.15},
            'source': 'EPA Stormwater BMP Database; 환경부 불투수면 저감 가이드라인',
            'study_count': 20,
        },
    },
    'rooftop_garden': {
        'name': '옥상정원',
        'name_en': 'Rooftop Garden',
        'icon': '🌻',
        'description': '옥상 녹화 및 텃밭 조성',
        'affects': ['greenRate', 'heatIsland', 'heatDays'],
        'base_effect': 1.0,  # v51: 3.0 → 1.0 (30㎡ 단위 효과)
        'effects': {'time': -0.12, 'space': 0.22, 'context': 0.15},
        'quantitative_effects': {
            'temperature_reduction': 5,
            'carbon_absorption': 15,
            'runoff_reduction': 60,
            'energy_saving': 600,
        },
        'cost': 150,
        'install_area': 30,
        'evidence': {
            'level': 'high',
            'confidence': 0.88,
            'effect_range': {'min': 0.8, 'max': 1.2},
            'source': 'Nature Communications Green Infrastructure 2024; 서울시 옥상녹화 효과분석',
            'study_count': 25,
        },
    },
    'shade_awning': {
        'name': '그늘막',
        'name_en': 'Shade Awning',
        'icon': '⛱️',
        'description': '건물 외부 그늘 조성',
        'affects': ['shelterAccess', 'elderlyRatio'],
        'base_effect': 0.5,  # v51: 1.5 → 0.5 (체감온도 저감 효과)
        'effects': {'time': -0.15, 'space': 0.05, 'context': 0.12},
        'quantitative_effects': {
            'temperature_reduction': 8,
            'carbon_absorption': 0,
            'runoff_reduction': 0,
            'energy_saving': 200,
        },
        'cost': 60,
        'install_area': 15,
        'evidence': {
            'level': 'medium',
            'confidence': 0.70,
            'effect_range': {'min': 0.6, 'max': 1.4},
            'source': 'Urban Acupuncture Framework 2024; 폭염대응 그늘막 효과 연구',
            'study_count': 6,
        },
    },
}

# ============================================
# 🔥 형평성(Equity) 지표 - IPCC AR6 Climate Justice
# 참고: Columbia Climate School UHI Equity Study
# ============================================
EQUITY_INDICATORS = {
    "low_income_ratio": {
        "name": "저소득층 비율",
        "weight": 0.30,
        "threshold": {"low": 10, "medium": 20, "high": 30},
        "description": "기초생활수급자 및 차상위계층 비율",
        "higher_is_worse": True,
    },
    "ac_penetration": {
        "name": "에어컨 보급률",
        "weight": 0.25,
        "threshold": {"low": 80, "medium": 60, "high": 40},
        "description": "가구당 냉방기기 보급률",
        "higher_is_worse": False,  # 낮을수록 취약
    },
    "outdoor_worker_ratio": {
        "name": "야외노동자 비율",
        "weight": 0.25,
        "threshold": {"low": 5, "medium": 10, "high": 20},
        "description": "건설/배달/농업 등 야외 근로자 비율",
        "higher_is_worse": True,
    },
    "single_elderly_ratio": {
        "name": "독거노인 비율",
        "weight": 0.20,
        "threshold": {"low": 5, "medium": 10, "high": 15},
        "description": "65세 이상 1인가구 비율",
        "higher_is_worse": True,
    },
}

# ============================================
# 🔥 게이미피케이션 배지 시스템
# 참고: Ant Forest, GoBeEco 프로젝트
# ============================================
BADGES = {
    "first_action": {
        "name": "첫 발걸음",
        "name_en": "First Step",
        "icon": "🌱",
        "description": "첫 번째 DIY 실천",
        "condition": {"total_diy_count": 1},
    },
    "green_starter": {
        "name": "그린 스타터",
        "name_en": "Green Starter",
        "icon": "🌿",
        "description": "DIY 3개 이상 실천",
        "condition": {"total_diy_count": 3},
    },
    "climate_guardian": {
        "name": "기후 수호자",
        "name_en": "Climate Guardian",
        "icon": "🛡️",
        "description": "DIY 10개 이상 실천",
        "condition": {"total_diy_count": 10},
    },
    "carbon_saver": {
        "name": "탄소 절약왕",
        "name_en": "Carbon Saver",
        "icon": "💨",
        "description": "누적 CO2 감축 100kg 달성",
        "condition": {"total_carbon_saved": 100},
    },
    "heat_fighter": {
        "name": "폭염 파이터",
        "name_en": "Heat Fighter",
        "icon": "🔥",
        "description": "쿨루프 또는 그린커튼 설치",
        "condition": {"diy_types": ["cool_roof", "green_curtain"]},
    },
    "flood_defender": {
        "name": "침수 방어자",
        "name_en": "Flood Defender",
        "icon": "🌊",
        "description": "빗물화분 또는 투수블록 설치",
        "condition": {"diy_types": ["rain_planter", "permeable_block"]},
    },
    "community_leader": {
        "name": "마을 리더",
        "name_en": "Community Leader",
        "icon": "👑",
        "description": "지역 상위 10% 기여도",
        "condition": {"regional_rank_percentile": 10},
    },
    "streak_7": {
        "name": "일주일 연속",
        "name_en": "Week Streak",
        "icon": "🔥",
        "description": "7일 연속 활동",
        "condition": {"streak_days": 7},
    },
}

# ============================================
# 🔥 climate_logic.py 동기화: 위기 점수 계산 기준치
# ============================================
CRISIS_THRESHOLDS = {
    "time": {
        "heat_days": {"low": 10, "medium": 20, "high": 30, "weight": 0.40},
        "flood_frequency": {"low": 1, "medium": 3, "high": 5, "weight": 0.35},
        "disaster_history": {"low": 0, "medium": 2, "high": 5, "weight": 0.25},
    },
    "space": {
        "impervious_rate": {"low": 30, "medium": 50, "high": 70, "weight": 0.35},
        "surface_temp": {"low": 2, "medium": 4, "high": 6, "weight": 0.35},
        "green_rate": {"low": 30, "medium": 15, "high": 10, "weight": 0.30, "inverse": True},
    },
    "context": {
        "shelter_access": {"low": 500, "medium": 1000, "high": 1500, "weight": 0.35, "inverse": True},
        "drainage_capacity": {"low": 80, "medium": 50, "high": 30, "weight": 0.35, "inverse": True},
        "elderly_ratio": {"low": 10, "medium": 20, "high": 30, "weight": 0.30},
    },
}

# ============================================
# 🔥 climate_logic.py 동기화: 정책 ROI 계산 상수
# ============================================
POLICY_ROI_CONSTANTS = {
    "carbon_price": 50,           # 원/kgCO2
    "electricity_price": 120,     # 원/kWh
    "heat_damage_cost": 5000,     # 원/일/인
    "flood_damage_cost": 150000,  # 원/㎡
    "green_benefit": 3000,        # 원/㎡/년
    "avg_diy_per_citizen": {
        "carbon_reduction": 12,       # kgCO2/년
        "temperature_reduction": 0.5, # °C
        "runoff_reduction": 5,        # ㎥/년
    },
}

# ============================================
# 유틸리티 함수
# ============================================

def load_logs():
    """로그 파일 읽기"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] 로그 파일 손상, 초기화: {e}")
            return []
    return []

def save_log(log_entry):
    """로그 추가 저장 (스레드 안전)"""
    with log_lock:
        try:
            logs = load_logs()
            logs.append(log_entry)
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            print(f"[LOG] 데이터 저장 완료. 총 {len(logs)}개 레코드")
        except Exception as e:
            print(f"[ERROR] 로그 저장 실패: {e}")


# ============================================
# 🔥 climate_logic.py 동기화: 점수 계산 함수
# ============================================

def calculate_axis_score(value, threshold, inverse=False):
    """
    축별 점수 계산 (0-100)
    climate_logic.py의 calculate_axis_score 함수와 동일
    """
    low = threshold["low"]
    medium = threshold["medium"]
    high = threshold["high"]

    if inverse:
        # 낮을수록 위험 (예: 녹지율, 배수용량)
        if value >= low:
            return 100
        elif value >= medium:
            return 50 + 50 * (value - medium) / (low - medium)
        elif value >= high:
            return 50 * (value - high) / (medium - high)
        else:
            return 0
    else:
        # 높을수록 위험 (예: 폭염일수, 불투수율)
        if value <= low:
            return 100
        elif value <= medium:
            return 50 + 50 * (medium - value) / (medium - low)
        elif value <= high:
            return 50 * (high - value) / (high - medium)
        else:
            return 0


def calculate_resilience_score_from_raw(raw_data):
    """
    3축 복원력 점수 계산 (climate_logic.py 동기화)

    raw_data: {
        'heat_days': int,
        'flood_frequency': int,
        'disaster_history': int,
        'impervious_rate': float,
        'surface_temp': float,  # heat_island
        'green_rate': float,
        'shelter_access': float,
        'drainage_capacity': float,
        'elderly_ratio': float,
    }
    """
    # 시간축 점수
    time_scores = []
    for key in ["heat_days", "flood_frequency", "disaster_history"]:
        threshold = CRISIS_THRESHOLDS["time"][key]
        value = raw_data.get(key, 0)
        score = calculate_axis_score(value, threshold)
        time_scores.append(score * threshold["weight"])
    time_score = sum(time_scores)

    # 공간축 점수
    space_scores = []
    for key in ["impervious_rate", "surface_temp", "green_rate"]:
        threshold = CRISIS_THRESHOLDS["space"][key]
        value = raw_data.get(key, 0)
        inverse = threshold.get("inverse", False)
        score = calculate_axis_score(value, threshold, inverse)
        space_scores.append(score * threshold["weight"])
    space_score = sum(space_scores)

    # 맥락축 점수
    context_scores = []
    for key in ["shelter_access", "drainage_capacity", "elderly_ratio"]:
        threshold = CRISIS_THRESHOLDS["context"][key]
        value = raw_data.get(key, 0)
        inverse = threshold.get("inverse", False)
        score = calculate_axis_score(value, threshold, inverse)
        context_scores.append(score * threshold["weight"])
    context_score = sum(context_scores)

    # 종합 점수
    total = (
        time_score * AXIS_WEIGHTS_CLIMATE["time"] +
        space_score * AXIS_WEIGHTS_CLIMATE["space"] +
        context_score * AXIS_WEIGHTS_CLIMATE["context"]
    )

    return {
        "time": round(time_score, 2),
        "space": round(space_score, 2),
        "context": round(context_score, 2),
        "total": round(total, 2),
        "resilience": round(total, 2),
    }


def apply_diy_effect_to_scores(scores, diy_id, count=1):
    """
    DIY 효과 적용 (climate_logic.py 동기화)

    수확체감의 법칙: sqrt(count)로 효과 감소
    """
    if diy_id not in DIY_ITEMS:
        return scores

    item = DIY_ITEMS[diy_id]
    effects = item.get('effects', {})

    # 수확체감의 법칙 적용
    effect_multiplier = np.sqrt(count)

    # 축별 효과 적용
    new_time = min(100, scores['time'] * (1 - effects.get('time', 0) * effect_multiplier))
    new_space = min(100, scores['space'] * (1 + effects.get('space', 0) * effect_multiplier))
    new_context = min(100, scores['context'] * (1 + effects.get('context', 0) * effect_multiplier))

    new_total = (
        new_time * AXIS_WEIGHTS_CLIMATE["time"] +
        new_space * AXIS_WEIGHTS_CLIMATE["space"] +
        new_context * AXIS_WEIGHTS_CLIMATE["context"]
    )

    return {
        "time": round(new_time, 2),
        "space": round(new_space, 2),
        "context": round(new_context, 2),
        "total": round(new_total, 2),
        "resilience": round(new_total, 2),
    }


def calculate_policy_roi(participants, applied_diy_list):
    """
    정책 ROI 계산 (climate_logic.py 동기화)

    participants: int - 참여자 수
    applied_diy_list: [{'diy_id': str, 'count': int}, ...]
    """
    avg = POLICY_ROI_CONSTANTS["avg_diy_per_citizen"]

    carbon_reduction = participants * avg["carbon_reduction"]
    temp_reduction = participants * avg["temperature_reduction"] / 1000  # 도시 평균 기여

    # 경제적 편익 계산
    carbon_benefit = carbon_reduction * POLICY_ROI_CONSTANTS["carbon_price"]

    # DIY별 에너지 절감량 계산
    total_energy_saving = 0
    for applied in applied_diy_list:
        diy_id = applied.get('diy_id')
        count = applied.get('count', 1)
        if diy_id in DIY_ITEMS:
            energy = DIY_ITEMS[diy_id].get('quantitative_effects', {}).get('energy_saving', 0)
            total_energy_saving += energy * count

    energy_benefit = total_energy_saving * POLICY_ROI_CONSTANTS["electricity_price"]

    economic_benefit = carbon_benefit + energy_benefit

    return {
        "total_participants": participants,
        "estimated_carbon_reduction": round(carbon_reduction, 2),  # kgCO2/년
        "estimated_temp_reduction": round(temp_reduction, 4),      # °C
        "estimated_flood_risk_reduction": round(participants * 0.01, 2),  # %
        "economic_benefit": round(economic_benefit, 0),  # 원/년
        "carbon_benefit": round(carbon_benefit, 0),
        "energy_benefit": round(energy_benefit, 0),
        "total_energy_saving": round(total_energy_saving, 0),  # kWh/년
        "social_benefit": f"시민 {participants:,}명이 참여하여 연간 {carbon_reduction:,.0f}kg CO2 감축 예상",
    }


def recommend_diy_for_region(raw_data, population_density=15000):
    """
    지역 특성에 맞는 DIY 추천 (climate_logic.py 동기화)
    """
    scores = {}

    for diy_id, item in DIY_ITEMS.items():
        score = 0
        reasons = []

        if diy_id == "cool_roof":
            if raw_data.get('surface_temp', 0) > 4:
                score += 30
                reasons.append("높은 표면온도")
            if raw_data.get('heat_days', 0) > 20:
                score += 20
                reasons.append("폭염일수 많음")

        elif diy_id == "rain_planter":
            if raw_data.get('flood_frequency', 0) > 2:
                score += 25
                reasons.append("침수 위험")
            if raw_data.get('green_rate', 100) < 20:
                score += 20
                reasons.append("녹지 부족")

        elif diy_id == "green_curtain":
            if raw_data.get('heat_days', 0) > 18:
                score += 25
                reasons.append("폭염 대비")
            if raw_data.get('surface_temp', 0) > 3:
                score += 20
                reasons.append("온도 저감 필요")

        elif diy_id == "permeable_block":
            if raw_data.get('impervious_rate', 0) > 60:
                score += 35
                reasons.append("높은 불투수율")
            if raw_data.get('flood_frequency', 0) > 1:
                score += 15
                reasons.append("배수 개선")

        elif diy_id == "rooftop_garden":
            if raw_data.get('heat_days', 0) > 15 and raw_data.get('green_rate', 100) < 25:
                score += 30
                reasons.append("폭염+녹지부족")
            if population_density > 15000:
                score += 15
                reasons.append("고밀도 지역")

        elif diy_id == "shade_awning":
            if raw_data.get('heat_days', 0) > 20:
                score += 25
                reasons.append("폭염 대비")
            if raw_data.get('elderly_ratio', 0) > 15:
                score += 20
                reasons.append("고령자 보호")

        scores[diy_id] = {"score": score, "reasons": reasons}

    # 점수순 정렬
    sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

    return {
        "top_id": sorted_items[0][0],
        "top_name": DIY_ITEMS[sorted_items[0][0]]['name'],
        "top_reason": ", ".join(sorted_items[0][1]["reasons"]) if sorted_items[0][1]["reasons"] else "기본 추천",
        "scores": scores,
        "rankings": [{"diy_id": k, "diy_name": DIY_ITEMS[k]['name'], "score": v["score"], "reasons": v["reasons"]} for k, v in sorted_items],
    }


# ============================================
# LightGBM 모델 학습/로드
# ============================================

def generate_synthetic_training_data(n_samples=500):
    """
    🔥 v51: climate_logic.ts 동기화된 합성 학습 데이터 생성

    [반영된 로직]
    1. IPCC AR6 비선형 리스크 모델 (Cascading Risk Amplification)
    2. 복합 위험(Compound Risk) 시너지
    3. 인프라 감쇄 (방어 모델)
    4. Huff Model 거리 감쇄 (800m 생활권)
    """
    np.random.seed(42)

    # 8개 지표 생성 (현실적인 범위)
    heat_days = np.random.uniform(15, 35, n_samples)  # 폭염일수: 15-35일
    flood_frequency = np.random.uniform(0, 5, n_samples)  # 침수빈도: 0-5회
    impervious_rate = np.random.uniform(30, 80, n_samples)  # 불투수율: 30-80%
    green_rate = np.random.uniform(10, 50, n_samples)  # 녹지율: 10-50%
    heat_island = np.random.uniform(1, 6, n_samples)  # 열섬지수: 1-6
    elderly_ratio = np.random.uniform(8, 25, n_samples)  # 고령자비율: 8-25%
    shelter_access = np.random.uniform(50, 800, n_samples)  # 쉼터거리: 50-800m
    medical_access = np.random.uniform(40, 95, n_samples)  # 의료접근성: 40-95%

    X = np.column_stack([
        heat_days, flood_frequency, impervious_rate, green_rate,
        heat_island, elderly_ratio, shelter_access, medical_access
    ])

    # ============================================
    # 🔥 v51: climate_logic.ts 동기화 복원력 계산
    # ============================================
    scores = []

    for i in range(n_samples):
        hd = heat_days[i]
        ff = flood_frequency[i]
        ir = impervious_rate[i]
        gr = green_rate[i]
        hi = heat_island[i]
        er = elderly_ratio[i]
        sa = shelter_access[i]
        ma = medical_access[i]

        # 1. 3축 위험 점수 계산 (0-100)
        # 시간축: 폭염 60% + 침수 40%
        heat_risk = min(100, (hd - 15) / 20 * 100)  # 15-35일 → 0-100
        flood_risk = min(100, ff / 5 * 100)  # 0-5회 → 0-100
        time_risk = heat_risk * 0.6 + flood_risk * 0.4

        # 공간축: 불투수 40% + 녹지부족 30% + 열섬 30%
        impervious_risk = min(100, (ir - 30) / 50 * 100)  # 30-80% → 0-100
        green_deficit_risk = min(100, (50 - gr) / 40 * 100)  # 녹지 50% 기준
        heat_island_risk = min(100, (hi - 1) / 5 * 100)  # 1-6 → 0-100
        space_risk = impervious_risk * 0.4 + green_deficit_risk * 0.3 + heat_island_risk * 0.3

        # 맥락축: 고령자 위험
        vuln_risk = min(100, (er - 8) / 17 * 100)  # 8-25% → 0-100

        # ============================================
        # 2. 🔥 복합 위험(Compound Risk) 시너지
        # ============================================
        compound_bonus = 0

        # 복합 위험 1: 도시 열파 (폭염 + 열섬 + 불투수)
        if hd >= 25 and hi >= 4.0 and ir >= 60:
            space_risk = min(100, space_risk * 1.25)
            compound_bonus += 5

        # 복합 위험 2: 도시 홍수 (침수 + 불투수 + 녹지부족)
        if ff >= 3 and ir >= 65 and gr <= 25:
            space_risk = min(100, space_risk * 1.30)
            compound_bonus += 7

        # 복합 위험 3: 취약계층 열파 (폭염 + 고령자 + 쉼터부족)
        if hd >= 25 and er >= 20 and sa >= 700:
            vuln_risk = min(100, vuln_risk * 1.35)
            compound_bonus += 8

        # ============================================
        # 3. 🔥 IPCC AR6 비선형 리스크 증폭
        # ============================================
        hazard_norm = time_risk / 100
        exposure_norm = space_risk / 100
        vuln_norm = vuln_risk / 100

        # Cascading Risk Amplification
        HAZARD_THRESHOLD = 0.4
        amplification_factor = 1.0
        if hazard_norm > HAZARD_THRESHOLD:
            excess = hazard_norm - HAZARD_THRESHOLD
            amplification_factor = 1 + 3.0 * (excess ** 2.5)

        # Exposure-Vulnerability 상호작용
        exposure_vuln_interaction = np.sqrt(exposure_norm * vuln_norm)

        # 비선형 리스크
        nonlinear_risk = hazard_norm * ((1 + exposure_vuln_interaction) ** amplification_factor)
        scaled_nonlinear_risk = min(100, nonlinear_risk * 65)

        # 선형 리스크 (가중 합산)
        linear_risk = time_risk * 0.35 + space_risk * 0.35 + vuln_risk * 0.30

        # 블렌딩 (비선형 60% + 선형 40%)
        base_risk = scaled_nonlinear_risk * 0.6 + linear_risk * 0.4

        # 복합 위험 보너스 추가
        base_risk = min(100, base_risk + compound_bonus)

        # ============================================
        # 4. 🔥 인프라 감쇄 (방어 모델)
        # ============================================
        # Huff Model: 800m 이내 100%, 초과 시 감쇄
        if sa <= 800:
            shelter_score = 100
        else:
            excess_dist = sa - 800
            shelter_score = 100 * np.exp(-excess_dist / 500)

        # 인프라 점수 (쉼터 + 의료)
        infra_score = shelter_score * 0.6 + ma * 0.4

        # 인프라 방어율 (최대 60%)
        defense_rate = min(0.6, infra_score / 200)

        # 취약성 저항 (고령자 비율이 높으면 인프라 효율 감소)
        vuln_resistance = (vuln_risk / 100) * 0.30
        final_defense = defense_rate * (1 - vuln_resistance)

        # 완화된 리스크
        mitigated_risk = base_risk * (1 - final_defense)

        # ============================================
        # 5. 최종 복원력 점수
        # ============================================
        resilience = 100 - mitigated_risk
        resilience = max(15, min(100, resilience))  # 15-100 범위

        scores.append(resilience)

    y = np.array(scores)

    # 노이즈 추가
    noise = np.random.normal(0, 3, n_samples)
    y = np.clip(y + noise, 0, 100)

    return X, y

def train_lightgbm_model():
    """LightGBM 모델 학습"""
    global model, shap_explainer, model_trained

    print("[ML] LightGBM 모델 학습 시작...")

    # 합성 데이터 생성
    X, y = generate_synthetic_training_data(n_samples=1000)

    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM 데이터셋
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 모델 파라미터
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # 모델 학습
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
    )

    # SHAP Explainer 생성
    print("[ML] SHAP Explainer 생성 중...")
    shap_explainer = shap.TreeExplainer(model)

    # 모델 저장
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SHAP_EXPLAINER_FILE, 'wb') as f:
        pickle.dump(shap_explainer, f)

    model_trained = True
    print("[ML] [OK] LightGBM 모델 학습 완료!")
    print(f"[ML] 검증 RMSE: {model.best_score['valid_0']['rmse']:.2f}")

def load_model():
    """저장된 모델 로드"""
    global model, shap_explainer, model_trained, diy_effect_model, diy_model_trained

    if os.path.exists(MODEL_FILE) and os.path.exists(SHAP_EXPLAINER_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            with open(SHAP_EXPLAINER_FILE, 'rb') as f:
                shap_explainer = pickle.load(f)
            model_trained = True
            print("[ML] [OK] 저장된 복원력 모델 로드 완료!")
        except Exception as e:
            print(f"[ML] 복원력 모델 로드 실패: {e}")

    # DIY 효과 모델 로드
    if os.path.exists(DIY_EFFECT_MODEL_FILE):
        try:
            with open(DIY_EFFECT_MODEL_FILE, 'rb') as f:
                diy_effect_model = pickle.load(f)
            diy_model_trained = True
            print("[ML] [OK] 저장된 DIY 효과 모델 로드 완료!")
        except Exception as e:
            print(f"[ML] DIY 효과 모델 로드 실패: {e}")

    return model_trained


# ============================================
# 🔥 DIY 효과 예측 모델 학습
# ============================================

def generate_diy_training_data(n_samples=2000):
    """
    DIY 효과 예측용 합성 데이터 생성

    X: [8개 지표, DIY 종류 (one-hot 6개), DIY 개수]
    y: 점수 변화량 (양수 = 개선)

    도메인 지식 반영:
    - DIY가 영향 주는 지표가 나쁠수록 효과 큼
    - DIY 개수에 따른 체감효과 (수확체감의 법칙)
    """
    np.random.seed(42)

    X_list = []
    y_list = []

    diy_ids = list(DIY_ITEMS.keys())

    for _ in range(n_samples):
        # 8개 지표 랜덤 생성
        heat_days = np.random.uniform(15, 35)
        flood_frequency = np.random.uniform(0, 5)
        impervious_rate = np.random.uniform(30, 80)
        green_rate = np.random.uniform(10, 50)
        heat_island = np.random.uniform(1, 6)
        elderly_ratio = np.random.uniform(8, 25)
        shelter_access = np.random.uniform(50, 800)
        medical_access = np.random.uniform(40, 95)

        features = [heat_days, flood_frequency, impervious_rate, green_rate,
                    heat_island, elderly_ratio, shelter_access, medical_access]

        # 랜덤 DIY 선택
        diy_id = np.random.choice(diy_ids)
        diy_info = DIY_ITEMS[diy_id]
        diy_one_hot = [1 if d == diy_id else 0 for d in diy_ids]

        # DIY 개수 (1-10개)
        diy_count = np.random.randint(1, 11)

        # X: 지표 8개 + DIY one-hot 6개 + 개수 1개 = 15개
        X_row = features + diy_one_hot + [diy_count]
        X_list.append(X_row)

        # 효과 계산 (도메인 지식 기반)
        base_effect = diy_info['base_effect']
        affects = diy_info['affects']

        # 해당 DIY가 영향 주는 지표의 "나쁜 정도"에 비례한 효과
        problem_severity = 0
        feature_map = {
            'heatDays': (heat_days - 22.4) / 22.4,  # 평균 대비 초과율
            'floodFrequency': (flood_frequency - 3.2) / 3.2,
            'imperviousRate': (impervious_rate - 55) / 55,
            'greenRate': (38 - green_rate) / 38,  # 녹지율은 반대
            'heatIsland': (heat_island - 3.2) / 3.2,
            'elderlyRatio': (elderly_ratio - 16.8) / 16.8,
            'shelterAccess': (shelter_access - 650) / 650,
            'medicalAccess': (66 - medical_access) / 66,  # 의료접근성은 반대
        }

        for affect in affects:
            if affect in feature_map:
                severity = max(0, feature_map[affect])  # 음수면 0 (이미 양호)
                problem_severity += severity

        # 효과 = 기본효과 × 문제심각도 × 개수효과 × 노이즈
        # 개수효과: sqrt(count) - 수확체감
        count_effect = np.sqrt(diy_count)
        effect = base_effect * (1 + problem_severity) * count_effect

        # 노이즈 추가
        noise = np.random.normal(0, 0.5)
        effect = max(0, effect + noise)  # 최소 0

        y_list.append(effect)

    return np.array(X_list), np.array(y_list)


def train_diy_effect_model():
    """DIY 효과 예측 모델 학습"""
    global diy_effect_model, diy_model_trained

    print("[ML] DIY effect model training started...")

    # 합성 데이터 생성
    X, y = generate_diy_training_data(n_samples=3000)

    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM 데이터셋
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 모델 파라미터
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    # 모델 학습
    diy_effect_model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
    )

    # 모델 저장
    with open(DIY_EFFECT_MODEL_FILE, 'wb') as f:
        pickle.dump(diy_effect_model, f)

    diy_model_trained = True
    print("[ML] [OK] DIY 효과 모델 학습 완료!")
    print(f"[ML] 검증 RMSE: {diy_effect_model.best_score['valid_0']['rmse']:.2f}")

# ============================================
# API 엔드포인트
# ============================================

@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "ok", "service": "GG Climate ML Server"})

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "model_status": "trained" if model_trained else "not_trained",
        "model_type": "LightGBM + SHAP",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    복원력 점수 예측 API (실제 LightGBM 모델 사용)
    """
    global model, model_trained

    try:
        raw_data = request.get_data(as_text=True)
        print(f"[DEBUG] /api/predict 요청 본문: {raw_data[:200] if raw_data else '(empty)'}")

        if not raw_data or raw_data.strip() == '':
            return jsonify({"error": "빈 요청 본문"}), 400

        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "JSON 파싱 실패"}), 400

        features = data.get('features', [])

        if len(features) != 8:
            return jsonify({
                "error": "8개 지표가 필요합니다",
                "received": len(features)
            }), 400

        # 모델이 없으면 학습
        if not model_trained:
            train_lightgbm_model()

        # LightGBM 예측
        X = np.array(features).reshape(1, -1)
        prediction = float(model.predict(X)[0])
        prediction = max(0, min(100, prediction))  # 0-100 범위 제한

        # 로그 저장
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": "/api/predict",
            "features": {
                "heatDays": features[0],
                "floodFrequency": features[1],
                "imperviousRate": features[2],
                "greenRate": features[3],
                "heatIsland": features[4],
                "elderlyRatio": features[5],
                "shelterAccess": features[6],
                "medicalAccess": features[7]
            },
            "prediction": round(prediction, 2)
        }
        save_log(log_entry)

        return jsonify({
            "prediction": round(prediction, 2),
            "confidence": 0.85,
            "model_version": "LightGBM-v1.0",
            "model_type": "LightGBM"
        })

    except Exception as e:
        print(f"[ERROR] /api/predict 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================
# 경기도 평균 기준값 (climate_logic.ts 동기화)
# ============================================
GYEONGGI_BASELINE = {
    'heatDays': 22.4,
    'floodFrequency': 3.2,
    'imperviousRate': 55.0,
    'greenRate': 38.0,
    'heatIsland': 3.2,
    'elderlyRatio': 16.8,
    'shelterAccess': 650,
    'medicalAccess': 66.0,
}

# 8개 지표 이름 (순서 중요)
FEATURE_NAMES = [
    'heatDays', 'floodFrequency', 'imperviousRate', 'greenRate',
    'heatIsland', 'elderlyRatio', 'shelterAccess', 'medicalAccess'
]

FEATURE_LABELS = [
    '폭염일수', '침수빈도', '불투수율', '녹지율',
    '열섬지수', '고령자비율', '쉼터접근성', '의료접근성'
]

# 지표별 방향성 (True = 높을수록 나쁨, False = 높을수록 좋음)
HIGHER_IS_WORSE = [True, True, True, False, True, True, True, False]

# 축별 가중치 (climate_logic.ts 동기화)
AXIS_WEIGHTS = [0.35, 0.35, 0.35, 0.35, 0.35, 0.30, 0.30, 0.30]  # time, time, space, space, space, context, context, context
INTRA_WEIGHTS = [0.6, 0.4, 0.35, 0.35, 0.30, 0.30, 0.35, 0.35]

# ============================================
# 🔥 v50: IPCC AR6 + VESTAP 기반 기후 리스크 Tier 가중치
# ============================================
# 선행연구 기반 (IPCC AR6, ND-GAIN, 한국 VESTAP):
# - 취약성 = (기후노출 × α) + (민감도 × β) - (적응역량 × γ)
# - 기후노출(Hazard/Exposure)이 최우선, 적응역량은 보조적
#
# Tier 1: 기후노출 (직접적 기후 위험) - 최우선
# Tier 2: 민감도 (취약성 악화 요인) - 높음
# Tier 3: 적응역량 (대응 능력) - 보조적
# ============================================
CLIMATE_RISK_TIER = {
    # Tier 1: 기후노출 (Hazard/Exposure) - 직접적 기후 위험
    'heatDays': {'tier': 1, 'weight': 1.0, 'category': 'hazard'},
    'floodFrequency': {'tier': 1, 'weight': 1.0, 'category': 'hazard'},
    'heatIsland': {'tier': 1, 'weight': 0.9, 'category': 'hazard'},

    # Tier 2: 민감도 (Sensitivity) - 취약성 악화 요인
    'imperviousRate': {'tier': 2, 'weight': 0.7, 'category': 'sensitivity'},
    'greenRate': {'tier': 2, 'weight': 0.7, 'category': 'sensitivity'},

    # Tier 3: 적응역량 (Adaptive Capacity) - 대응 능력
    'elderlyRatio': {'tier': 3, 'weight': 0.5, 'category': 'adaptive_capacity'},
    'shelterAccess': {'tier': 3, 'weight': 0.4, 'category': 'adaptive_capacity'},
    'medicalAccess': {'tier': 3, 'weight': 0.3, 'category': 'adaptive_capacity'},
}

def calculate_climate_risk_score(indicator):
    """
    🔥 v50: 기후 리스크 점수 계산 (IPCC 기반)

    점수 = |shap_value| × tier_weight × (1 + gap_severity)

    - tier_weight: 기후노출 > 민감도 > 적응역량
    - gap_severity: gap이 클수록 추가 가중치
    """
    feature_name = indicator.get('feature_name', '')
    tier_info = CLIMATE_RISK_TIER.get(feature_name, {'tier': 3, 'weight': 0.5})

    shap_value = abs(indicator.get('shap_value', 0))
    gap_percent = abs(indicator.get('gap_percent', 0))

    # gap이 20% 이상이면 추가 보너스 (심각한 문제)
    gap_severity = min(0.5, gap_percent / 40)  # 최대 50% 보너스

    # 최종 점수 = SHAP × Tier 가중치 × (1 + gap 심각도)
    risk_score = shap_value * tier_info['weight'] * (1 + gap_severity)

    return {
        'risk_score': risk_score,
        'tier': tier_info['tier'],
        'tier_weight': tier_info['weight'],
        'category': tier_info['category'],
    }

def calculate_gap_based_shap(features):
    """
    Gap 기반 SHAP 값 계산 (climate_logic.ts의 Gap 분석과 동일)

    - Gap = (현재값 - 평균) / 평균 * 100
    - SHAP = Gap * 축 가중치 * 축내 가중치 * 방향성
    - 정렬: |SHAP| 절대값 큰 순 (AI가 분석한 영향력 순위)

    🔥 v43_direction_fix:
    - is_problem: 실제 문제인지 여부 (녹지율 높음 = 좋은 것 = 문제 아님)
    - 높을수록 나쁜 지표: 평균보다 높으면 문제
    - 높을수록 좋은 지표: 평균보다 낮으면 문제
    """
    shap_details = []
    baseline_keys = list(GYEONGGI_BASELINE.keys())

    for i in range(8):
        value = features[i]
        baseline = GYEONGGI_BASELINE[baseline_keys[i]]

        # Gap 계산: (현재값 - 평균) / 평균 * 100
        gap = ((value - baseline) / baseline) * 100

        # SHAP 값 계산
        if HIGHER_IS_WORSE[i]:
            # 높을수록 나쁨: gap이 양수면 음수 영향 (복원력 감소)
            shap_value = -gap * AXIS_WEIGHTS[i] * INTRA_WEIGHTS[i]
            # 🔥 문제 판단: 평균보다 높으면 문제
            is_problem = gap > 0
        else:
            # 높을수록 좋음: gap이 양수면 양수 영향 (복원력 증가)
            shap_value = gap * AXIS_WEIGHTS[i] * INTRA_WEIGHTS[i]
            # 🔥 문제 판단: 평균보다 낮으면 문제 (녹지율 낮음 = 문제)
            is_problem = gap < 0

        shap_details.append({
            'feature_name': FEATURE_NAMES[i],
            'feature_label': FEATURE_LABELS[i],
            'feature_value': value,
            'baseline': baseline,
            'gap_percent': round(gap, 2),
            'shap_value': round(shap_value, 4),
            'impact': 'positive' if shap_value >= 0 else 'negative',
            'is_problem': is_problem,  # 🔥 실제 문제 여부
            'higher_is_worse': HIGHER_IS_WORSE[i],  # 🔥 방향성 정보
        })

    # |SHAP| 절대값 큰 순으로 정렬 (AI가 분석한 영향력 순위)
    shap_details.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    # 순위 부여
    for i, item in enumerate(shap_details):
        item['importance_rank'] = i + 1

    return shap_details

@app.route('/api/shap', methods=['POST'])
def get_shap_values():
    """
    🔥 v44_full_ml: 실제 LightGBM TreeSHAP + Gap 기반 하이브리드

    하이브리드 비율:
    - 데이터 < 100건: Gap 70% + TreeSHAP 30%
    - 데이터 100-500건: Gap 50% + TreeSHAP 50%
    - 데이터 > 500건: Gap 30% + TreeSHAP 70%

    TreeSHAP의 장점:
    - 비선형 관계 발견 (임계점 등)
    - 지표 간 상호작용 포착
    - 데이터에서 학습한 실제 패턴 반영
    """
    global model, shap_explainer, model_trained

    try:
        raw_data = request.get_data(as_text=True)
        print(f"[DEBUG] /api/shap 요청 본문: {raw_data[:200] if raw_data else '(empty)'}")

        if not raw_data or raw_data.strip() == '':
            return jsonify({"error": "빈 요청 본문"}), 400

        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "JSON 파싱 실패"}), 400

        features = data.get('features', [])

        if len(features) == 7:
            features.append(features[6])
        elif len(features) != 8:
            return jsonify({
                "error": "7개 또는 8개 지표가 필요합니다",
                "received": len(features)
            }), 400

        # 모델이 없으면 학습
        if not model_trained:
            train_lightgbm_model()

        # 1. Gap 기반 SHAP 계산
        gap_shap_details = calculate_gap_based_shap(features)

        # 2. 실제 LightGBM TreeSHAP 계산
        X = np.array(features).reshape(1, -1)
        lgb_shap_values = shap_explainer.shap_values(X)
        lgb_shap_list = lgb_shap_values[0].tolist()

        # 3. 🔥 하이브리드 비율 결정 (데이터 양 기반)
        logs = load_logs()
        data_count = len([l for l in logs if l.get('endpoint') == '/api/predict'])

        if data_count < 100:
            gap_weight, ml_weight = 0.7, 0.3
        elif data_count < 500:
            gap_weight, ml_weight = 0.5, 0.5
        else:
            gap_weight, ml_weight = 0.3, 0.7

        print(f"[SHAP] 하이브리드 비율: Gap {gap_weight*100:.0f}% + ML {ml_weight*100:.0f}% (데이터 {data_count}건)")

        # 4. 하이브리드 SHAP 값 계산
        hybrid_shap_details = []
        for i, name in enumerate(FEATURE_NAMES):
            gap_item = next((d for d in gap_shap_details if d['feature_name'] == name), None)
            gap_shap = gap_item['shap_value'] if gap_item else 0
            ml_shap = lgb_shap_list[i]

            # 하이브리드 SHAP
            hybrid_shap = gap_shap * gap_weight + ml_shap * ml_weight

            # is_problem 판단 (Gap 기반 - 통계적 기준)
            is_problem = gap_item.get('is_problem', False) if gap_item else False

            hybrid_shap_details.append({
                'feature_name': name,
                'feature_label': FEATURE_LABELS[i],
                'feature_value': features[i],
                'baseline': GYEONGGI_BASELINE[name],
                'gap_percent': gap_item['gap_percent'] if gap_item else 0,
                'gap_shap': round(gap_shap, 4),
                'ml_shap': round(ml_shap, 4),
                'shap_value': round(hybrid_shap, 4),  # 🔥 하이브리드
                'impact': 'positive' if hybrid_shap >= 0 else 'negative',
                'is_problem': is_problem,
                'higher_is_worse': HIGHER_IS_WORSE[i],
            })

        # |SHAP| 절대값 큰 순으로 정렬
        hybrid_shap_details.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        for i, item in enumerate(hybrid_shap_details):
            item['importance_rank'] = i + 1

        # shap_values 배열 (원래 순서)
        shap_values_ordered = []
        for name in FEATURE_NAMES:
            item = next((d for d in hybrid_shap_details if d['feature_name'] == name), None)
            shap_values_ordered.append(item['shap_value'] if item else 0)

        # 🔥 v50: top_risk 결정 - IPCC 기반 Tier 가중치 적용
        # 단순 |shap_value| 대신 기후 리스크 중요도 반영
        problem_indicators = [d for d in hybrid_shap_details if d.get('is_problem', False)]

        if problem_indicators:
            # 각 문제 지표에 기후 리스크 점수 추가
            for indicator in problem_indicators:
                risk_info = calculate_climate_risk_score(indicator)
                indicator['climate_risk_score'] = risk_info['risk_score']
                indicator['tier'] = risk_info['tier']
                indicator['tier_weight'] = risk_info['tier_weight']
                indicator['category'] = risk_info['category']

            # 기후 리스크 점수 기준 정렬 (Tier 가중치 반영)
            # 1순위: Tier (낮을수록 우선)
            # 2순위: climate_risk_score (높을수록 우선)
            problem_indicators.sort(key=lambda x: (x.get('tier', 3), -x.get('climate_risk_score', 0)))

            top_risk = problem_indicators[0]

            # 디버그 로그
            print(f"[SHAP v50] top_risk 결정:")
            for p in problem_indicators[:3]:
                print(f"  - {p['feature_label']}: tier={p.get('tier')}, risk_score={p.get('climate_risk_score', 0):.2f}, shap={abs(p['shap_value']):.2f}")
        else:
            top_risk = None

        # 로그 저장
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": "/api/shap",
            "features": {FEATURE_NAMES[i]: features[i] for i in range(8)},
            "hybrid_shap_values": shap_values_ordered,
            "gap_weight": gap_weight,
            "ml_weight": ml_weight,
            "top_risk": top_risk,
            "problem_count": len(problem_indicators)
        }
        save_log(log_entry)

        return jsonify({
            "shap_values": shap_values_ordered,
            "shap_details": hybrid_shap_details,
            "lgb_shap_values": lgb_shap_list,
            "top_risk": top_risk,
            "problem_indicators": problem_indicators,
            "feature_names": FEATURE_LABELS,
            "hybrid_weights": {"gap": gap_weight, "ml": ml_weight},
            "data_count": data_count,
            "model_type": "Hybrid (Gap + LightGBM TreeSHAP)",
            "model_version": "v44_full_ml"
        })

    except Exception as e:
        print(f"[ERROR] /api/shap 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/diy-effect', methods=['POST'])
def predict_diy_effect():
    """
    🔥 v44_full_ml: DIY 효과 예측 API

    입력:
    - features: 8개 지표
    - diy_id: DIY 종류 (cool_roof, rain_planter, green_curtain, permeable_block, rooftop_garden, shade_awning)
    - diy_count: DIY 개수 (기본값: 1)

    출력:
    - predicted_effect: 예상 점수 변화량
    - confidence: 예측 신뢰도
    - explanation: 효과 설명
    """
    global diy_effect_model, diy_model_trained

    try:
        data = request.get_json(force=True)
        features = data.get('features', [])
        diy_id = data.get('diy_id', 'cool_roof')
        diy_count = data.get('diy_count', 1)

        if len(features) != 8:
            return jsonify({"error": "8개 지표가 필요합니다"}), 400

        if diy_id not in DIY_ITEMS:
            return jsonify({"error": f"알 수 없는 DIY: {diy_id}", "valid_diy": list(DIY_ITEMS.keys())}), 400

        # 모델이 없으면 학습
        if not diy_model_trained:
            train_diy_effect_model()

        # DIY one-hot 인코딩
        diy_ids = list(DIY_ITEMS.keys())
        diy_one_hot = [1 if d == diy_id else 0 for d in diy_ids]

        # X: 지표 8개 + DIY one-hot 6개 + 개수 1개 = 15개
        X = np.array(features + diy_one_hot + [diy_count]).reshape(1, -1)

        # 예측
        predicted_effect = float(diy_effect_model.predict(X)[0])
        predicted_effect = max(0, predicted_effect)  # 최소 0

        # DIY 정보
        diy_info = DIY_ITEMS[diy_id]

        # 효과 설명 생성
        if predicted_effect >= 3:
            effect_level = "매우 효과적"
            confidence = 0.9
        elif predicted_effect >= 2:
            effect_level = "효과적"
            confidence = 0.8
        elif predicted_effect >= 1:
            effect_level = "보통"
            confidence = 0.7
        else:
            effect_level = "효과 적음"
            confidence = 0.6

        explanation = f"{diy_info['name']} {diy_count}개 설치 시 기후 안전점수 +{predicted_effect:.1f}점 예상 ({effect_level})"

        return jsonify({
            "diy_id": diy_id,
            "diy_name": diy_info['name'],
            "diy_count": diy_count,
            "predicted_effect": round(predicted_effect, 2),
            "confidence": confidence,
            "effect_level": effect_level,
            "explanation": explanation,
            "affects": diy_info['affects'],
            "model_version": "v44_full_ml"
        })

    except Exception as e:
        print(f"[ERROR] /api/diy-effect 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/diy-recommendations', methods=['POST'])
def get_diy_recommendations():
    """
    🔥 v49: SHAP 문제 지표 매칭 가중치 적용

    선행연구 기반 (ND-GAIN, IPCC AR6, EPA Climate Adaptation):
    - Gap-to-Solution Matching: 가장 큰 Gap을 해결하는 솔루션 우선
    - 문제 심각도 × 해결 적합성 가중치

    가중치 공식:
    - 최종 점수 = 기본 효과 × (1 + 문제 매칭 보너스)
    - top_risk 매칭: +50%
    - problem_indicators 상위 3개 매칭: +30%
    - 문제 심각도(gap%) 비례 보너스: +gap% × 0.3

    입력:
    - features: 8개 지표
    - diy_count: DIY 개수 (기본값: 1)

    출력:
    - recommendations: DIY별 예상 효과 (문제 매칭 가중치 적용 후 정렬)
    """
    global diy_effect_model, diy_model_trained

    try:
        data = request.get_json(force=True)
        features = data.get('features', [])
        diy_count = data.get('diy_count', 1)

        if len(features) != 8:
            return jsonify({"error": "8개 지표가 필요합니다"}), 400

        # 모델이 없으면 학습
        if not diy_model_trained:
            train_diy_effect_model()

        # 🔥 v50: IPCC Tier 기반 문제 지표 파악
        shap_details = calculate_gap_based_shap(features)
        problem_indicators = [d for d in shap_details if d['is_problem']]

        # 각 문제 지표에 Tier 정보 추가
        for indicator in problem_indicators:
            risk_info = calculate_climate_risk_score(indicator)
            indicator['tier'] = risk_info['tier']
            indicator['climate_risk_score'] = risk_info['risk_score']
            indicator['category'] = risk_info['category']

        # Tier 기준 정렬 (기후노출 > 민감도 > 적응역량)
        problem_indicators.sort(key=lambda x: (x.get('tier', 3), -x.get('climate_risk_score', 0)))

        # top_risk: Tier 기반 가장 심각한 문제 지표
        top_risk = problem_indicators[0] if problem_indicators else None
        top_risk_name = top_risk['feature_name'] if top_risk else None
        top_risk_gap = abs(top_risk['gap_percent']) if top_risk else 0
        top_risk_tier = top_risk.get('tier', 3) if top_risk else 3

        # 상위 3개 문제 지표 (Tier 기준)
        top3_problem_names = [d['feature_name'] for d in problem_indicators[:3]]

        print(f"[DIY-REC v50] top_risk: {top_risk_name} (tier={top_risk_tier}, gap: {top_risk_gap:.1f}%), top3: {top3_problem_names}")

        diy_ids = list(DIY_ITEMS.keys())
        recommendations = []

        for diy_id in diy_ids:
            diy_info = DIY_ITEMS[diy_id]
            diy_one_hot = [1 if d == diy_id else 0 for d in diy_ids]

            X = np.array(features + diy_one_hot + [diy_count]).reshape(1, -1)
            base_effect = float(diy_effect_model.predict(X)[0])
            base_effect = max(0, base_effect)

            # 🔥 v49b: 문제 매칭 가중치 강화 (선행연구: 문제 해결 적합성 최우선)
            affects = diy_info['affects']
            bonus = 0.0
            match_reason = []

            # top_risk 매칭: +100% (핵심 문제 해결 DIY 우선)
            if top_risk_name and top_risk_name in affects:
                bonus += 1.0
                match_reason.append(f"top_risk({top_risk_name})")
                # 추가: 문제 심각도 비례 보너스 (+gap% × 0.5, 최대 50%)
                gap_bonus = min(0.5, top_risk_gap * 0.005)
                bonus += gap_bonus

            # top3 problem 매칭: +40% (top_risk 제외)
            for prob_name in top3_problem_names:
                if prob_name != top_risk_name and prob_name in affects:
                    bonus += 0.4
                    match_reason.append(f"top3({prob_name})")
                    break  # 중복 방지

            # 최종 점수 계산
            final_effect = base_effect * (1 + bonus)

            recommendations.append({
                "diy_id": diy_id,
                "diy_name": diy_info['name'],
                "base_effect": round(base_effect, 2),
                "predicted_effect": round(final_effect, 2),
                "bonus": round(bonus * 100, 1),  # 보너스 %
                "match_reason": match_reason,
                "affects": affects,
            })

        # 🔥 v49b: 정렬 - top_risk 매칭 DIY 우선, 그 다음 점수순
        # 선행연구(IPCC AR6): "가장 시급한 문제를 해결하는 적응 옵션이 최우선"
        def sort_key(rec):
            has_top_risk = 'top_risk' in str(rec.get('match_reason', []))
            return (
                0 if has_top_risk else 1,  # top_risk 매칭 DIY 우선
                -rec['predicted_effect']   # 그 다음 효과 높은 순
            )
        recommendations.sort(key=sort_key)

        # 순위 부여
        for i, rec in enumerate(recommendations):
            rec['rank'] = i + 1

        return jsonify({
            "diy_count": diy_count,
            "recommendations": recommendations,
            "top_recommendation": recommendations[0] if recommendations else None,
            "top_risk": top_risk_name,
            "problem_indicators": top3_problem_names,
            "model_version": "v49_shap_matching"
        })

    except Exception as e:
        print(f"[ERROR] /api/diy-recommendations 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """모델 재학습 API"""
    try:
        train_lightgbm_model()
        return jsonify({
            "status": "success",
            "message": "모델 학습 완료",
            "model_type": "LightGBM"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/stats', methods=['GET'])
def get_log_stats():
    """수집된 데이터 통계"""
    logs = load_logs()

    if not logs:
        return jsonify({
            "total_records": 0,
            "message": "수집된 데이터가 없습니다"
        })

    prediction_logs = [log for log in logs if log.get('endpoint') == '/api/predict']

    if not prediction_logs:
        return jsonify({
            "total_records": len(logs),
            "prediction_records": 0
        })

    avg_prediction = np.mean([log['prediction'] for log in prediction_logs])

    features_avg = {}
    for key in ["heatDays", "floodFrequency", "imperviousRate", "greenRate",
                "heatIsland", "elderlyRatio", "shelterAccess", "medicalAccess"]:
        values = [log['features'][key] for log in prediction_logs if key in log.get('features', {})]
        if values:
            features_avg[key] = round(np.mean(values), 2)

    return jsonify({
        "total_records": len(logs),
        "prediction_records": len(prediction_logs),
        "avg_prediction": round(avg_prediction, 2),
        "features_avg": features_avg,
        "first_record": logs[0]['timestamp'],
        "last_record": logs[-1]['timestamp'],
        "model_trained": model_trained
    })

@app.route('/api/logs/export', methods=['GET'])
def export_logs():
    """학습용 데이터 내보내기"""
    logs = load_logs()
    prediction_logs = [log for log in logs if log.get('endpoint') == '/api/predict']

    training_data = []
    for log in prediction_logs:
        if 'features' in log:
            training_data.append({
                "X": [
                    log['features']['heatDays'],
                    log['features']['floodFrequency'],
                    log['features']['imperviousRate'],
                    log['features']['greenRate'],
                    log['features']['heatIsland'],
                    log['features']['elderlyRatio'],
                    log['features']['shelterAccess'],
                    log['features']['medicalAccess']
                ],
                "y": log['prediction'],
                "timestamp": log['timestamp']
            })

    return jsonify({
        "count": len(training_data),
        "data": training_data
    })


# ============================================
# 🔥 climate_logic.py 동기화: 신규 API 엔드포인트
# ============================================

@app.route('/api/resilience-score', methods=['POST'])
def get_resilience_score():
    """
    🔥 climate_logic.py 동기화: 3축 복원력 점수 계산 API

    입력 (raw_data):
    - heat_days: 폭염일수
    - flood_frequency: 침수빈도
    - disaster_history: 재해이력 (선택)
    - impervious_rate: 불투수율
    - surface_temp: 표면온도/열섬지수
    - green_rate: 녹지율
    - shelter_access: 쉼터접근성 (거리, m)
    - drainage_capacity: 배수용량 (선택)
    - elderly_ratio: 고령자비율

    출력:
    - time: 시간축 점수
    - space: 공간축 점수
    - context: 맥락축 점수
    - total: 종합 점수
    - resilience: 복원력 점수
    """
    try:
        data = request.get_json(force=True)
        raw_data = data.get('raw_data', data)  # raw_data 키가 있으면 사용, 없으면 전체 사용

        # 필수 필드 확인
        required_fields = ['heat_days', 'impervious_rate', 'green_rate', 'elderly_ratio']
        for field in required_fields:
            if field not in raw_data:
                return jsonify({"error": f"필수 필드 누락: {field}"}), 400

        # 기본값 설정
        raw_data.setdefault('flood_frequency', 0)
        raw_data.setdefault('disaster_history', 0)
        raw_data.setdefault('surface_temp', raw_data.get('heat_island', 3.2))
        raw_data.setdefault('shelter_access', 650)
        raw_data.setdefault('drainage_capacity', 60)

        scores = calculate_resilience_score_from_raw(raw_data)

        return jsonify({
            "scores": scores,
            "raw_data": raw_data,
            "axis_weights": AXIS_WEIGHTS_CLIMATE,
            "model_version": "climate_logic_sync"
        })

    except Exception as e:
        print(f"[ERROR] /api/resilience-score 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/apply-diy', methods=['POST'])
def apply_diy():
    """
    🔥 climate_logic.py 동기화: DIY 효과 적용 API

    입력:
    - scores: 현재 점수 {time, space, context, total, resilience}
    - diy_id: DIY 종류
    - count: DIY 개수 (기본값: 1)

    출력:
    - original_scores: 원래 점수
    - new_scores: DIY 적용 후 점수
    - improvement: 개선량
    - diy_info: DIY 정보
    """
    try:
        data = request.get_json(force=True)
        scores = data.get('scores', {})
        diy_id = data.get('diy_id', 'cool_roof')
        count = data.get('count', 1)

        # 필수 필드 확인
        if not all(k in scores for k in ['time', 'space', 'context']):
            return jsonify({"error": "scores에 time, space, context 필드가 필요합니다"}), 400

        if diy_id not in DIY_ITEMS:
            return jsonify({"error": f"알 수 없는 DIY: {diy_id}", "valid_diy": list(DIY_ITEMS.keys())}), 400

        # total, resilience 기본값
        scores.setdefault('total', (
            scores['time'] * AXIS_WEIGHTS_CLIMATE["time"] +
            scores['space'] * AXIS_WEIGHTS_CLIMATE["space"] +
            scores['context'] * AXIS_WEIGHTS_CLIMATE["context"]
        ))
        scores.setdefault('resilience', scores['total'])

        new_scores = apply_diy_effect_to_scores(scores, diy_id, count)
        diy_info = DIY_ITEMS[diy_id]

        improvement = {
            "time": round(new_scores['time'] - scores['time'], 2),
            "space": round(new_scores['space'] - scores['space'], 2),
            "context": round(new_scores['context'] - scores['context'], 2),
            "total": round(new_scores['total'] - scores['total'], 2),
        }

        return jsonify({
            "original_scores": scores,
            "new_scores": new_scores,
            "improvement": improvement,
            "diy_id": diy_id,
            "diy_name": diy_info['name'],
            "diy_count": count,
            "diy_info": {
                "name": diy_info['name'],
                "name_en": diy_info['name_en'],
                "icon": diy_info['icon'],
                "description": diy_info['description'],
                "effects": diy_info['effects'],
                "quantitative_effects": diy_info['quantitative_effects'],
                "cost": diy_info['cost'],
                "install_area": diy_info['install_area'],
            },
            "model_version": "climate_logic_sync"
        })

    except Exception as e:
        print(f"[ERROR] /api/apply-diy 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/policy-roi', methods=['POST'])
def get_policy_roi():
    """
    🔥 climate_logic.py 동기화: 정책 ROI 계산 API

    입력:
    - participants: 참여자 수
    - applied_diy: [{'diy_id': str, 'count': int}, ...]

    출력:
    - total_participants: 총 참여자
    - estimated_carbon_reduction: 탄소 감축량 (kgCO2/년)
    - estimated_temp_reduction: 온도 저감 (°C)
    - economic_benefit: 경제적 편익 (원/년)
    - social_benefit: 사회적 편익 설명
    """
    try:
        data = request.get_json(force=True)
        participants = data.get('participants', 100)
        applied_diy = data.get('applied_diy', [])

        if not isinstance(participants, int) or participants < 0:
            return jsonify({"error": "participants는 0 이상의 정수여야 합니다"}), 400

        roi = calculate_policy_roi(participants, applied_diy)

        return jsonify({
            **roi,
            "constants": POLICY_ROI_CONSTANTS,
            "model_version": "climate_logic_sync"
        })

    except Exception as e:
        print(f"[ERROR] /api/policy-roi 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend-diy-by-region', methods=['POST'])
def recommend_diy_by_region():
    """
    🔥 climate_logic.py 동기화: 지역 특성 기반 DIY 추천 API

    입력 (raw_data):
    - heat_days: 폭염일수
    - flood_frequency: 침수빈도
    - impervious_rate: 불투수율
    - surface_temp: 표면온도/열섬지수
    - green_rate: 녹지율
    - elderly_ratio: 고령자비율
    - population_density: 인구밀도 (선택, 기본값: 15000)

    출력:
    - top_id: 1순위 DIY ID
    - top_name: 1순위 DIY 이름
    - top_reason: 추천 이유
    - rankings: 전체 순위
    """
    try:
        data = request.get_json(force=True)
        raw_data = data.get('raw_data', data)
        population_density = data.get('population_density', 15000)

        recommendation = recommend_diy_for_region(raw_data, population_density)

        return jsonify({
            **recommendation,
            "population_density": population_density,
            "model_version": "climate_logic_sync"
        })

    except Exception as e:
        print(f"[ERROR] /api/recommend-diy-by-region 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/diy-items', methods=['GET'])
def get_diy_items():
    """
    🔥 climate_logic.py 동기화: DIY 아이템 목록 API

    모든 DIY 아이템 정보 반환
    """
    items = []
    for diy_id, info in DIY_ITEMS.items():
        items.append({
            "id": diy_id,
            "name": info['name'],
            "name_en": info['name_en'],
            "icon": info['icon'],
            "description": info['description'],
            "affects": info['affects'],
            "base_effect": info['base_effect'],
            "effects": info['effects'],
            "quantitative_effects": info['quantitative_effects'],
            "cost": info['cost'],
            "install_area": info['install_area'],
        })

    return jsonify({
        "items": items,
        "count": len(items),
        "axis_weights": AXIS_WEIGHTS_CLIMATE,
        "crisis_thresholds": CRISIS_THRESHOLDS,
        "policy_roi_constants": POLICY_ROI_CONSTANTS,
        "model_version": "climate_logic_sync"
    })


@app.route('/api/crisis-thresholds', methods=['GET'])
def get_crisis_thresholds():
    """
    🔥 climate_logic.py 동기화: 위기 기준치 API

    모든 축별 위기 기준치 반환
    """
    return jsonify({
        "thresholds": CRISIS_THRESHOLDS,
        "axis_weights": AXIS_WEIGHTS_CLIMATE,
        "model_version": "climate_logic_sync"
    })


# ============================================
# 🔥 게이미피케이션 API (시민 참여 플랫폼)
# 참고: Ant Forest, GoBeEco 프로젝트
# ============================================

@app.route('/api/citizen-impact', methods=['POST'])
def get_citizen_impact():
    """
    🔥 시민 기후행동 기여도 계산 API

    입력:
    - citizen_id: 시민 ID (선택)
    - applied_diy: [{'diy_id': str, 'count': int, 'installed_at': str}, ...]
    - region_id: 지역 ID (선택)

    출력:
    - total_carbon_saved: 누적 CO2 감축량 (kg)
    - total_temperature_reduction: 누적 온도 저감 기여 (°C)
    - total_runoff_reduction: 누적 우수 저감량 (L)
    - total_energy_saved: 누적 에너지 절감량 (kWh)
    - level: 시민 레벨 (1-10)
    - earned_badges: 획득 배지 목록
    - next_badge: 다음 목표 배지
    """
    try:
        data = request.get_json(force=True)
        applied_diy = data.get('applied_diy', [])
        citizen_id = data.get('citizen_id', 'anonymous')
        region_id = data.get('region_id', 'unknown')

        # 누적 효과 계산
        total_carbon = 0
        total_temp = 0
        total_runoff = 0
        total_energy = 0
        diy_type_set = set()
        total_diy_count = 0

        for item in applied_diy:
            diy_id = item.get('diy_id')
            count = item.get('count', 1)

            if diy_id in DIY_ITEMS:
                diy_info = DIY_ITEMS[diy_id]
                qe = diy_info.get('quantitative_effects', {})

                # 수확체감 법칙 적용 (sqrt)
                effect_mult = np.sqrt(count)

                total_carbon += qe.get('carbon_absorption', 0) * effect_mult
                total_temp += qe.get('temperature_reduction', 0) * effect_mult * 0.01  # 도시 기여도
                total_runoff += qe.get('runoff_reduction', 0) * effect_mult
                total_energy += qe.get('energy_saving', 0) * effect_mult

                diy_type_set.add(diy_id)
                total_diy_count += count

        # 레벨 계산 (로그 스케일)
        level = min(10, max(1, int(np.log2(total_diy_count + 1)) + 1))

        # 배지 판정
        earned_badges = []
        next_badge = None

        for badge_id, badge_info in BADGES.items():
            condition = badge_info.get('condition', {})
            earned = False

            if 'total_diy_count' in condition:
                if total_diy_count >= condition['total_diy_count']:
                    earned = True
            elif 'total_carbon_saved' in condition:
                if total_carbon >= condition['total_carbon_saved']:
                    earned = True
            elif 'diy_types' in condition:
                if any(dt in diy_type_set for dt in condition['diy_types']):
                    earned = True

            if earned:
                earned_badges.append({
                    "id": badge_id,
                    "name": badge_info['name'],
                    "name_en": badge_info['name_en'],
                    "icon": badge_info['icon'],
                    "description": badge_info['description'],
                })
            elif next_badge is None:
                # 아직 획득하지 못한 첫 번째 배지를 다음 목표로
                next_badge = {
                    "id": badge_id,
                    "name": badge_info['name'],
                    "icon": badge_info['icon'],
                    "condition": condition,
                }

        # 경제적 가치 환산
        economic_value = (
            total_carbon * POLICY_ROI_CONSTANTS['carbon_price'] +
            total_energy * POLICY_ROI_CONSTANTS['electricity_price']
        )

        return jsonify({
            "citizen_id": citizen_id,
            "region_id": region_id,
            "total_diy_count": total_diy_count,
            "diy_types_count": len(diy_type_set),
            "impact": {
                "carbon_saved_kg": round(total_carbon, 2),
                "temperature_reduction_c": round(total_temp, 4),
                "runoff_reduction_l": round(total_runoff, 2),
                "energy_saved_kwh": round(total_energy, 2),
                "economic_value_krw": round(economic_value, 0),
            },
            "level": level,
            "level_name": ["새싹", "풀잎", "나무", "숲", "정원", "공원", "마을", "도시", "지구", "우주"][level - 1],
            "earned_badges": earned_badges,
            "badge_count": len(earned_badges),
            "next_badge": next_badge,
            "model_version": "gamification_v1"
        })

    except Exception as e:
        print(f"[ERROR] /api/citizen-impact 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/community-leaderboard', methods=['POST'])
def get_community_leaderboard():
    """
    🔥 지역 커뮤니티 리더보드 API

    입력:
    - region_id: 지역 ID
    - citizens: [{'citizen_id': str, 'total_carbon_saved': float, 'total_diy_count': int}, ...]

    출력:
    - leaderboard: 순위별 시민 목록
    - region_stats: 지역 전체 통계
    """
    try:
        data = request.get_json(force=True)
        region_id = data.get('region_id', 'unknown')
        citizens = data.get('citizens', [])

        if not citizens:
            return jsonify({
                "region_id": region_id,
                "leaderboard": [],
                "region_stats": {"total_participants": 0}
            })

        # 탄소 감축량 기준 정렬
        sorted_citizens = sorted(citizens, key=lambda x: x.get('total_carbon_saved', 0), reverse=True)

        # 순위 부여 및 백분위 계산
        total_count = len(sorted_citizens)
        leaderboard = []

        for i, citizen in enumerate(sorted_citizens[:20]):  # 상위 20명만
            rank = i + 1
            percentile = round((1 - rank / total_count) * 100, 1)

            leaderboard.append({
                "rank": rank,
                "citizen_id": citizen.get('citizen_id'),
                "total_carbon_saved": citizen.get('total_carbon_saved', 0),
                "total_diy_count": citizen.get('total_diy_count', 0),
                "percentile": percentile,
                "is_top_10_percent": percentile >= 90,
            })

        # 지역 전체 통계
        total_carbon = sum(c.get('total_carbon_saved', 0) for c in citizens)
        total_diy = sum(c.get('total_diy_count', 0) for c in citizens)
        avg_carbon = total_carbon / total_count if total_count > 0 else 0

        return jsonify({
            "region_id": region_id,
            "leaderboard": leaderboard,
            "region_stats": {
                "total_participants": total_count,
                "total_carbon_saved": round(total_carbon, 2),
                "total_diy_count": total_diy,
                "avg_carbon_per_citizen": round(avg_carbon, 2),
            },
            "model_version": "gamification_v1"
        })

    except Exception as e:
        print(f"[ERROR] /api/community-leaderboard 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/badges', methods=['GET'])
def get_badges():
    """
    🔥 배지 목록 API
    """
    badges_list = []
    for badge_id, badge_info in BADGES.items():
        badges_list.append({
            "id": badge_id,
            "name": badge_info['name'],
            "name_en": badge_info['name_en'],
            "icon": badge_info['icon'],
            "description": badge_info['description'],
            "condition": badge_info['condition'],
        })

    return jsonify({
        "badges": badges_list,
        "count": len(badges_list),
        "model_version": "gamification_v1"
    })


# ============================================
# 🔥 형평성(Equity) 평가 API
# 참고: IPCC AR6 Climate Justice, Columbia UHI Study
# ============================================

@app.route('/api/equity-score', methods=['POST'])
def get_equity_score():
    """
    🔥 형평성 취약 점수 계산 API

    입력 (equity_data):
    - low_income_ratio: 저소득층 비율 (%)
    - ac_penetration: 에어컨 보급률 (%)
    - outdoor_worker_ratio: 야외노동자 비율 (%)
    - single_elderly_ratio: 독거노인 비율 (%)

    출력:
    - equity_score: 형평성 점수 (0-100, 높을수록 취약)
    - vulnerability_level: 취약 수준 (low/medium/high/critical)
    - priority_groups: 우선 지원 대상 그룹
    - recommendations: 맞춤형 정책 권고
    """
    try:
        data = request.get_json(force=True)
        equity_data = data.get('equity_data', data)

        # 각 지표별 취약 점수 계산 (0-100)
        scores = {}
        total_weighted_score = 0

        for indicator_id, indicator_info in EQUITY_INDICATORS.items():
            value = equity_data.get(indicator_id, 0)
            threshold = indicator_info['threshold']
            weight = indicator_info['weight']
            higher_is_worse = indicator_info['higher_is_worse']

            # 점수 계산 (0-100)
            if higher_is_worse:
                # 높을수록 취약 (예: 저소득층 비율)
                if value <= threshold['low']:
                    score = (value / threshold['low']) * 33
                elif value <= threshold['medium']:
                    score = 33 + ((value - threshold['low']) / (threshold['medium'] - threshold['low'])) * 33
                elif value <= threshold['high']:
                    score = 66 + ((value - threshold['medium']) / (threshold['high'] - threshold['medium'])) * 34
                else:
                    score = 100
            else:
                # 낮을수록 취약 (예: 에어컨 보급률)
                if value >= threshold['low']:
                    score = (1 - value / 100) * 33
                elif value >= threshold['medium']:
                    score = 33 + ((threshold['low'] - value) / (threshold['low'] - threshold['medium'])) * 33
                elif value >= threshold['high']:
                    score = 66 + ((threshold['medium'] - value) / (threshold['medium'] - threshold['high'])) * 34
                else:
                    score = 100

            score = max(0, min(100, score))
            scores[indicator_id] = {
                "name": indicator_info['name'],
                "value": value,
                "score": round(score, 2),
                "weight": weight,
            }
            total_weighted_score += score * weight

        # 종합 형평성 점수
        equity_score = round(total_weighted_score, 2)

        # 취약 수준 판정
        if equity_score < 25:
            vulnerability_level = "low"
            level_name = "양호"
        elif equity_score < 50:
            vulnerability_level = "medium"
            level_name = "주의"
        elif equity_score < 75:
            vulnerability_level = "high"
            level_name = "취약"
        else:
            vulnerability_level = "critical"
            level_name = "매우 취약"

        # 우선 지원 대상 그룹 (점수 50 이상)
        priority_groups = []
        for indicator_id, score_info in scores.items():
            if score_info['score'] >= 50:
                priority_groups.append({
                    "group": EQUITY_INDICATORS[indicator_id]['name'],
                    "score": score_info['score'],
                    "description": EQUITY_INDICATORS[indicator_id]['description'],
                })

        # 맞춤형 정책 권고
        recommendations = []
        if equity_data.get('low_income_ratio', 0) > 20:
            recommendations.append({
                "target": "저소득층",
                "action": "DIY 비용 지원 및 무료 설치 프로그램",
                "priority": "high"
            })
        if equity_data.get('ac_penetration', 100) < 60:
            recommendations.append({
                "target": "냉방 취약가구",
                "action": "무더위 쉼터 접근성 개선 및 그늘막 우선 설치",
                "priority": "high"
            })
        if equity_data.get('outdoor_worker_ratio', 0) > 10:
            recommendations.append({
                "target": "야외노동자",
                "action": "작업장 그늘막 설치 및 폭염 알림 서비스",
                "priority": "medium"
            })
        if equity_data.get('single_elderly_ratio', 0) > 10:
            recommendations.append({
                "target": "독거노인",
                "action": "방문 안전 확인 및 쿨루프 우선 설치",
                "priority": "high"
            })

        return jsonify({
            "equity_score": equity_score,
            "vulnerability_level": vulnerability_level,
            "vulnerability_level_name": level_name,
            "indicator_scores": scores,
            "priority_groups": priority_groups,
            "recommendations": recommendations,
            "equity_indicators": EQUITY_INDICATORS,
            "model_version": "equity_v1"
        })

    except Exception as e:
        print(f"[ERROR] /api/equity-score 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/equity-indicators', methods=['GET'])
def get_equity_indicators():
    """
    🔥 형평성 지표 목록 API
    """
    return jsonify({
        "indicators": EQUITY_INDICATORS,
        "count": len(EQUITY_INDICATORS),
        "source": "IPCC AR6 Climate Justice Framework; Columbia Climate School UHI Equity Study",
        "model_version": "equity_v1"
    })


# ============================================
# 서버 실행
# ============================================

# ============================================
# 앱 시작 시 모델 로드 (gunicorn 호환)
# ============================================
try:
    print("=" * 60)
    print("[ML Server] Gyeonggi Climate Platform ML API")
    print("=" * 60)

    # 저장된 모델 로드 시도
    if not load_model():
        print("[ML] No saved resilience model. Training now...")
        train_lightgbm_model()

    # DIY 효과 모델이 없으면 자동 학습
    if not diy_model_trained:
        print("[ML] No saved DIY effect model. Training now...")
        train_diy_effect_model()

    # 기존 로그 파일 확인
    if os.path.exists(LOG_FILE):
        logs = load_logs()
        print(f"[OK] Loaded {len(logs)} log records")
    else:
        print("[INFO] New log file will be created")

    print("=" * 60)
except Exception as e:
    print(f"[WARNING] Model loading failed: {e}")
    print("[ML Server] Starting without models (API will return fallback responses)")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
