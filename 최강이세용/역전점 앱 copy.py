import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# --- 페이지 설정 ---
st.set_page_config(page_title="투수 역전점 시뮬레이터", page_icon="⚾", layout="centered")

# --- 사이드바 스타일 ---
st.markdown("""
    <style>
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stSelectbox>div {
            padding: 0.5em;
        }
    </style>
""", unsafe_allow_html=True)

# --- 데이터 불러오기 및 전처리 ---
fatigue_df = pd.read_csv('피로도_with_index.csv', encoding='utf-8')
fatigue_df.drop(columns=['구속_변화량'], inplace=True)
fatigue_df['부상위험도'] = fatigue_df['부상위험도'].fillna(0)
fatigue_df['피로도지표'] = fatigue_df['피로도지수'] * 100
fatigue_df['구장'] = fatigue_df['구장'].map({'Home': 1, 'Away': 0})

# --- 외부환경변수 ---
env_vars = ['누적이동거리', '온도', '구장']
scaler = StandardScaler()
env_z = scaler.fit_transform(fatigue_df[env_vars])
env_z_df = pd.DataFrame(env_z, columns=[f"{col}_z" for col in env_vars])
fatigue_df = pd.concat([fatigue_df.reset_index(drop=True), env_z_df], axis=1)
fatigue_df['환경지수_z'] = env_z_df.mean(axis=1)
fatigue_df['외부환경변수'] = (
    0.433381 * fatigue_df['온도_z'] +
    0.437590 * fatigue_df['누적이동거리_z'] +
    0.129029 * fatigue_df['구장_z']
)

# --- 선수기량변수 ---
skill_vars = ['ERA', 'WHIP', '직구_피안타율']
skill_z = StandardScaler().fit_transform(fatigue_df[skill_vars])
skill_z_df = pd.DataFrame(skill_z, columns=[f"{col}_z" for col in skill_vars])
fatigue_df = pd.concat([fatigue_df.reset_index(drop=True), skill_z_df], axis=1)
fatigue_df['선수기량변수'] = (
    0.402767 * fatigue_df['ERA_z'] +
    0.503297 * fatigue_df['WHIP_z'] +
    0.093936 * fatigue_df['직구_피안타율_z']
)

# --- 신체조건변수 ---
phys_vars = ['나이', '키', '몸무게', '부상위험도']
phys_scaled = StandardScaler().fit_transform(fatigue_df[phys_vars])
fa = FactorAnalysis(n_components=1, random_state=42)
fatigue_df['신체조건변수'] = fa.fit_transform(phys_scaled)

# --- 앱 시작 ---
df = fatigue_df
st.title("⚾ 투수 역전점 시뮬레이터")
st.markdown("""
<style>
    .stApp {
        font-family: 'Apple SD Gothic Neo', sans-serif;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

투수1 = st.selectbox("👤 투수 A 선택", df['선수'].unique())
투수2 = st.selectbox("👤 투수 B 선택", df['선수'].unique())

투수1_dates = df[df['선수'] == 투수1]['날짜'].unique()
투수2_dates = df[df['선수'] == 투수2]['날짜'].unique()

날짜1 = st.selectbox(f"🗓️ {투수1}의 등판 날짜", 투수1_dates, key='날짜1')
날짜2 = st.selectbox(f"🗓️ {투수2}의 등판 날짜", 투수2_dates, key='날짜2')

if st.button("⚖️ 비교하기"):
    row_A = df[(df['선수'] == 투수1) & (df['날짜'] == 날짜1)].iloc[0]
    row_B = df[(df['선수'] == 투수2) & (df['날짜'] == 날짜2)].iloc[0]

    X_raw = df[['선수기량변수', '신체조건변수', '외부환경변수', '피로도지표']]
    y = df['WHIP']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    model = LinearRegression()
    model.fit(X_scaled, y)

    b0, b1, b2, b3, b4 = model.intercept_, *model.coef_

    a_scaled = scaler.transform([[row_A['선수기량변수'], row_A['신체조건변수'], row_A['외부환경변수'], 0]])[0]
    b_scaled = scaler.transform([[row_B['선수기량변수'], row_B['신체조건변수'], row_B['외부환경변수'], row_B['피로도지표']]])[0]

    numerator = b1 * (a_scaled[0] - b_scaled[0]) + b2 * (a_scaled[1] - b_scaled[1]) + b3 * (a_scaled[2] - b_scaled[2])
    denominator = b4
    역전점_standard = b_scaled[3] - (numerator / denominator)

    지표_평균 = df['피로도지표'].mean()
    지표_표준편차 = df['피로도지표'].std()
    역전점_피로도지표 = 역전점_standard * 지표_표준편차 + 지표_평균

    if 역전점_피로도지표 < 0:
        st.warning(f"❗ {투수1}은(는) 아무리 피로해져도 {투수2}보다 성능이 낮아지지 않습니다.")
    else:
        st.success(f"✅ 역전점 (피로도지표 기준): {투수1}이(가) 피로도지표 {역전점_피로도지표:.1f} 이상일 때 {투수2}보다 성능이 낮아집니다.")
        st.info(f"ℹ️ 현재 피로도 — {투수1}: {row_A['피로도지표']:.1f}, {투수2}: {row_B['피로도지표']:.1f}")

        if row_A['피로도지표'] < 역전점_피로도지표:
            st.success(f"🟢 따라서 현재는 {투수1}의 기용이 더 효율적입니다.")
        else:
            st.success(f"🔴 현재는 {투수2}의 기용이 더 효율적입니다.")
