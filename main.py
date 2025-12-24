import io
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Page / Font
# =========================
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

st.title("🌱 극지식물 최적 EC 농도 연구")


# =========================
# Constants
# =========================
SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]

TARGET_EC = {"송도고": 1.0, "하늘고": 2.0, "아라고": 4.0, "동산고": 8.0}
SCHOOL_COLOR = {"송도고": "#1f77b4", "하늘고": "#2ca02c", "아라고": "#ff7f0e", "동산고": "#d62728"}


# =========================
# NFC/NFD helpers
# =========================
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def contains_loose(text: str, keyword: str) -> bool:
    # NFC/NFD 양방향 포함 검사
    return (keyword in nfc(text)) or (keyword in nfd(text))


def safe_iterdir(p: Path):
    try:
        return list(p.iterdir())
    except Exception:
        return []


def iter_files_recursive(root: Path, max_depth: int = 5):
    """
    ✅ glob/rglob 없이 Path.iterdir()만으로 재귀 탐색
    - Streamlit Cloud 경로 차이/서브폴더 이슈에도 최대한 견고
    """
    if not root.exists():
        return

    stack = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        for child in safe_iterdir(cur):
            if child.is_file():
                yield child
            elif child.is_dir() and depth < max_depth:
                stack.append((child, depth + 1))


# =========================
# File detection (NO hardcoding)
# =========================
def detect_env_csv_files(data_dir: Path):
    """
    - data/ 내부에서 csv + 학교명 포함이면 매핑
    - (수정) 같은 파일명 변형(괄호 등)도 OK
    """
    mapping = {}
    if not data_dir.exists():
        return mapping

    for p in safe_iterdir(data_dir):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".csv":
            continue
        for school in SCHOOLS:
            if contains_loose(p.name, school):
                mapping[school] = p
                break
    return mapping


def detect_growth_xlsx_file(data_dir: Path, app_dir: Path):
    """
    ✅ 매우 강하게 탐색:
    1) data/에서 .xlsx 찾기
    2) 없으면 app_dir 아래를 iterdir 재귀탐색(= glob 금지 준수)
    3) 후보 여러 개면 '생육'/'결과' 점수로 우선순위
    """
    candidates = []

    # 1) data/ 우선
    if data_dir.exists():
        for p in safe_iterdir(data_dir):
            if p.is_file() and p.suffix.lower() == ".xlsx":
                score = 0
                if contains_loose(p.name, "생육"):
                    score += 2
                if contains_loose(p.name, "결과"):
                    score += 1
                candidates.append((score, p))

    # 2) 그래도 없으면 app_dir 전체 재귀 탐색
    if not candidates and app_dir.exists():
        for p in iter_files_recursive(app_dir, max_depth=6):
            if p.suffix.lower() == ".xlsx":
                score = 0
                if contains_loose(p.name, "생육"):
                    score += 2
                if contains_loose(p.name, "결과"):
                    score += 1
                candidates.append((score, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], nfc(x[1].name)))
    return candidates[0][1]


def detect_school_from_sheet(sheet_name: str):
    for school in SCHOOLS:
        if contains_loose(sheet_name, school):
            return school
    return None


# =========================
# Data loading (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_env_data(data_dir_str: str):
    data_dir = Path(data_dir_str)
    csv_map = detect_env_csv_files(data_dir)

    env_by_school = {}
    errors = {}
    for school, fp in csv_map.items():
        try:
            df = pd.read_csv(fp)

            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")

            for col in ["temperature", "humidity", "ph", "ec"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            env_by_school[school] = df
        except Exception as e:
            errors[school] = f"{fp.name}: {e}"

    return env_by_school, errors, {k: v.name for k, v in csv_map.items()}


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir_str: str, app_dir_str: str):
    data_dir = Path(data_dir_str)
    app_dir = Path(app_dir_str)

    xlsx_fp = detect_growth_xlsx_file(data_dir, app_dir)
    if xlsx_fp is None:
        # ✅ 에러로 죽이지 않고 "없음" 상태로 반환
        return {}, "생육 결과 XLSX 파일을 찾지 못했습니다. (앱은 계속 실행됩니다)", None

    try:
        sheet_dict = pd.read_excel(xlsx_fp, sheet_name=None, engine="openpyxl")
    except Exception as e:
        # ✅ 읽기 실패도 앱이 죽지 않도록
        return {}, f"XLSX 로딩 실패: {e} (앱은 계속 실행됩니다)", xlsx_fp

    growth_by_school = {}
    unknown_sheets = []

    for sheet_name, df in sheet_dict.items():
        for col in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        school = detect_school_from_sheet(sheet_name)
        if school is None:
            unknown_sheets.append(sheet_name)
        else:
            growth_by_school[school] = df

    warn_msg = None
    if unknown_sheets:
        warn_msg = "학교명을 포함하지 않는 시트가 있어(자동 매핑 불가) 제외했습니다: " + ", ".join(unknown_sheets[:10])

    return growth_by_school, warn_msg, xlsx_fp


def combine_env(env_by_school: dict, selected: str):
    if selected == "전체":
        frames = []
        for school, df in env_by_school.items():
            tmp = df.copy()
            tmp["school"] = school
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = env_by_school.get(selected, pd.DataFrame()).copy()
    if not df.empty:
        df["school"] = selected
    return df


def combine_growth(growth_by_school: dict, selected: str):
    if selected == "전체":
        frames = []
        for school, df in growth_by_school.items():
            tmp = df.copy()
            tmp["school"] = school
            tmp["target_ec"] = TARGET_EC.get(school)
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = growth_by_school.get(selected, pd.DataFrame()).copy()
    if not df.empty:
        df["school"] = selected
        df["target_ec"] = TARGET_EC.get(selected)
    return df


# =========================
# Paths (your structure)
# =========================
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

# =========================
# Sidebar
# =========================
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

# =========================
# Load data
# =========================
with st.spinner("데이터를 불러오는 중..."):
    env_by_school, env_errors, env_csv_detected = load_env_data(str(DATA_DIR))
    growth_by_school, growth_warn, growth_xlsx_fp = load_growth_data(str(DATA_DIR), str(APP_DIR))

# =========================
# Debug expander (필수: Cloud에서 확인)
# =========================
with st.expander("🔎 data/ 탐색 디버그(파일 인식 문제 해결용)", expanded=False):
    st.write("APP_DIR:", str(APP_DIR))
    st.write("DATA_DIR:", str(DATA_DIR))
    st.write("data/ 파일 목록:")
    for p in safe_iterdir(DATA_DIR):
        if p.is_file():
            st.write("-", p.name)

    st.write("✅ 감지된 환경 CSV:", env_csv_detected)
    st.write("✅ 감지된 생육 XLSX:", "-" if growth_xlsx_fp is None else str(growth_xlsx_fp))

# CSV 로딩 에러는 보여주되 앱은 진행
if env_errors:
    st.warning("일부 환경 CSV 로딩 중 문제가 있었습니다:\n- " + "\n- ".join([f"{k}: {v}" for k, v in env_errors.items()]))

# 생육 쪽 경고/안내도 "에러로 중단" 금지
if growth_warn:
    st.warning(growth_warn)


# =========================
# Prepare frames
# =========================
env_df = combine_env(env_by_school, selected_school)
growth_df = combine_growth(growth_by_school, selected_school)

# 환경 평균
env_means = []
for school in SCHOOLS:
    df = env_by_school.get(school)
    if df is None or df.empty:
        continue
    env_means.append(
        {
            "school": school,
            "temperature_mean": df["temperature"].mean() if "temperature" in df.columns else None,
            "humidity_mean": df["humidity"].mean() if "humidity" in df.columns else None,
            "ph_mean": df["ph"].mean() if "ph" in df.columns else None,
            "ec_mean": df["ec"].mean() if "ec" in df.columns else None,
            "target_ec": TARGET_EC.get(school),
        }
    )
env_means_df = pd.DataFrame(env_means)

# 생육 요약
growth_summ = []
for school in SCHOOLS:
    df = growth_by_school.get(school)
    if df is None or df.empty:
        continue
    growth_summ.append(
        {
            "school": school,
            "target_ec": TARGET_EC.get(school),
            "n": len(df),
            "mean_weight": df["생중량(g)"].mean() if "생중량(g)" in df.columns else None,
            "mean_leaf": df["잎 수(장)"].mean() if "잎 수(장)" in df.columns else None,
            "mean_shoot": df["지상부 길이(mm)"].mean() if "지상부 길이(mm)" in df.columns else None,
        }
    )
growth_summ_df = pd.DataFrame(growth_summ)


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# -------------------------
# Tab 1
# -------------------------
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
본 연구는 4개 학교(송도고, 하늘고, 아라고, 동산고)에서 서로 다른 목표 EC 조건으로 극지식물을 재배하며,
환경(온도/습도/pH/EC)과 생육 결과(생중량/잎수/길이)의 차이를 비교하여 **최적 EC 농도(특히 EC 2.0 조건의 유효성)**를 도출하는 것을 목표로 합니다.
"""
    )

    st.subheader("학교별 EC 조건")
    rows = []
    for school in SCHOOLS:
        n = int(growth_summ_df.loc[growth_summ_df["school"] == school, "n"].iloc[0]) if (not growth_summ_df.empty and (growth_summ_df["school"] == school).any()) else None
        rows.append({"학교명": school, "EC 목표": TARGET_EC.get(school), "개체수": n, "색상": SCHOOL_COLOR.get(school)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    total_n = int(growth_summ_df["n"].sum()) if (not growth_summ_df.empty and "n" in growth_summ_df.columns) else 0
    avg_temp = env_df["temperature"].mean() if (not env_df.empty and "temperature" in env_df.columns) else None
    avg_hum = env_df["humidity"].mean() if (not env_df.empty and "humidity" in env_df.columns) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n:,}")
    c2.metric("평균 온도", "-" if avg_temp is None else f"{avg_temp:.2f} ℃")
    c3.metric("평균 습도", "-" if avg_hum is None else f"{avg_hum:.2f} %")
    c4.metric("최적 EC", "2.0 (하늘고)")


# -------------------------
# Tab 2
# -------------------------
with tab2:
    st.subheader("학교별 환경 평균 비교 (2x2)")

    if env_means_df.empty:
        st.info("환경 평균 비교를 위한 데이터가 없습니다(환경 CSV를 확인하세요).")
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC"),
            horizontal_spacing=0.10,
            vertical_spacing=0.15,
        )

        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["temperature_mean"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in env_means_df["school"]]),
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["humidity_mean"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in env_means_df["school"]]),
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["ph_mean"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in env_means_df["school"]]),
            ),
            row=2, col=1
        )
        fig.add_trace(go.Bar(x=env_means_df["school"], y=env_means_df["target_ec"], name="목표 EC", marker=dict(opacity=0.75)), row=2, col=2)
        fig.add_trace(go.Bar(x=env_means_df["school"], y=env_means_df["ec_mean"], name="실측 평균 EC", marker=dict(opacity=0.75)), row=2, col=2)

        fig.update_layout(
            barmode="group",
            height=650,
            font=dict(family=PLOTLY_FONT_FAMILY),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    if selected_school == "전체":
        st.info("학교별 측정 주기가 달라서, 특정 학교 선택 시 더 명확합니다.")
    else:
        if env_df.empty:
            st.info("선택한 학교의 환경 데이터가 없습니다.")
        else:
            if "time" in env_df.columns:
                env_df = env_df.sort_values("time")

            if "temperature" in env_df.columns:
                fig_t = px.line(env_df, x="time" if "time" in env_df.columns else env_df.index, y="temperature", title="온도 변화")
                fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_t, use_container_width=True)

            if "humidity" in env_df.columns:
                fig_h = px.line(env_df, x="time" if "time" in env_df.columns else env_df.index, y="humidity", title="습도 변화")
                fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_h, use_container_width=True)

            if "ec" in env_df.columns:
                fig_ec = px.line(env_df, x="time" if "time" in env_df.columns else env_df.index, y="ec", title="EC 변화 (목표 EC 수평선 포함)")
                target = TARGET_EC.get(selected_school)
                if target is not None:
                    fig_ec.add_hline(y=target, line_dash="dash", annotation_text=f"목표 EC={target}", annotation_position="top left")
                fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_ec, use_container_width=True)

    st.divider()
    with st.expander("환경 데이터 원본 테이블 + CSV 다운로드"):
        if env_df.empty:
            st.info("표시할 환경 데이터가 없습니다.")
        else:
            st.dataframe(env_df, use_container_width=True)
            st.download_button(
                "⬇️ 환경 데이터 CSV 다운로드",
                data=env_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="환경데이터.csv",
                mime="text/csv",
            )


# -------------------------
# Tab 3 (NEVER crash)
# -------------------------
with tab3:
    if not growth_by_school:
        st.info(
            "생육 결과(XLSX)를 아직 읽지 못했어요. (앱은 정상 실행 중)\n\n"
            "1) 위의 🔎 디버그에서 '감지된 생육 XLSX'가 뜨는지 확인\n"
            "2) Streamlit Cloud가 최신 커밋을 배포했는지 확인"
        )
    else:
        st.subheader("🥇 핵심 결과 카드")

        tmp = growth_summ_df.dropna(subset=["mean_weight"]).copy()
        best_row = tmp.sort_values("mean_weight", ascending=False).head(1)

        best_mean_weight = None
        best_school = None
        best_ec = None
        if not best_row.empty:
            best_mean_weight = float(best_row["mean_weight"].iloc[0])
            best_school = str(best_row["school"].iloc[0])
            best_ec = float(best_row["target_ec"].iloc[0])

        sky_mean_weight = None
        if (growth_summ_df["school"] == "하늘고").any():
            sky_mean_weight = growth_summ_df.loc[growth_summ_df["school"] == "하늘고", "mean_weight"].iloc[0]

        c1, c2 = st.columns(2)
        c1.metric("EC별 평균 생중량(최대)", "-" if best_mean_weight is None else f"{best_mean_weight:.3f} g", help=f"{best_school} (EC {best_ec})")
        c2.metric("하늘고(EC 2.0) 평균 생중량(최적 강조)", "-" if pd.isna(sky_mean_weight) else f"{float(sky_mean_weight):.3f} g")

        st.divider()
        st.subheader("EC별 생육 비교 (2x2)")

        gplot = growth_summ_df.dropna(subset=["target_ec"]).copy().sort_values("target_ec")

        fig2 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량 (⭐)", "평균 잎 수", "평균 지상부 길이", "개체수 비교"),
            horizontal_spacing=0.10,
            vertical_spacing=0.15,
        )

        fig2.add_trace(go.Bar(x=gplot["target_ec"], y=gplot["mean_weight"], marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]])), row=1, col=1)
        fig2.add_trace(go.Bar(x=gplot["target_ec"], y=gplot["mean_leaf"], marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]])), row=1, col=2)
        fig2.add_trace(go.Bar(x=gplot["target_ec"], y=gplot["mean_shoot"], marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]])), row=2, col=1)
        fig2.add_trace(go.Bar(x=gplot["target_ec"], y=gplot["n"], marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]])), row=2, col=2)

        fig2.add_vline(x=2.0, line_dash="dash", annotation_text="하늘고 EC=2.0", annotation_position="top")
        fig2.update_layout(height=650, font=dict(family=PLOTLY_FONT_FAMILY), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("학교별 생중량 분포 (바이올린/박스)")

        if growth_df.empty or ("생중량(g)" not in growth_df.columns):
            st.info("생중량 분포를 표시할 데이터가 없습니다.")
        else:
            fig_dist = px.violin(
                growth_df.dropna(subset=["생중량(g)"]),
                x="school",
                y="생중량(g)",
                box=True,
                points="all",
                title="학교별 생중량 분포",
                category_orders={"school": SCHOOLS},
                color="school",
                color_discrete_map=SCHOOL_COLOR,
            )
            fig_dist.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()
        with st.expander("학교별 생육 데이터 원본 + XLSX 다운로드"):
            st.dataframe(growth_df, use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                if selected_school == "전체":
                    for school, df in growth_by_school.items():
                        df.to_excel(writer, sheet_name=school, index=False)
                else:
                    df_one = growth_by_school.get(selected_school, pd.DataFrame())
                    df_one.to_excel(writer, sheet_name=selected_school, index=False)

            buffer.seek(0)
            st.download_button(
                label="⬇️ 생육 데이터 XLSX 다운로드",
                data=buffer,
                file_name="생육데이터.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if growth_xlsx_fp is not None:
                st.caption(f"원본 생육 XLSX 감지: {growth_xlsx_fp}")


st.caption("© Polar Plant EC Dashboard • Streamlit + Plotly")
