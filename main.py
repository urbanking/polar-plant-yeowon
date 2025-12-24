import io
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Page / Font (Korean safe)
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
# Constants (no filename hardcoding)
# =========================
SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]

TARGET_EC = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적 (요구사항 강조)
    "아라고": 4.0,
    "동산고": 8.0,
}

SCHOOL_COLOR = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}


# =========================
# NFC/NFD helpers
# =========================
def norm_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm_nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def same_name_loose(a: str, b: str) -> bool:
    """
    NFC/NFD 양방향 비교로 '같은 파일명'을 최대한 안전하게 판단.
    """
    return (
        norm_nfc(a) == norm_nfc(b)
        or norm_nfd(a) == norm_nfd(b)
        or norm_nfc(a) == norm_nfd(b)
        or norm_nfd(a) == norm_nfc(b)
    )


def find_best_match_file(data_dir: Path, predicate):
    """
    Path.iterdir()로 파일을 순회하며 predicate를 만족하는 첫 파일 반환.
    (glob 패턴만 쓰는 방식 금지 대응)
    """
    if not data_dir.exists():
        return None

    for p in data_dir.iterdir():
        if p.is_file():
            try:
                if predicate(p):
                    return p
            except Exception:
                # predicate 내부 오류가 나도 앱이 죽지 않게 방어
                continue
    return None


def list_files(data_dir: Path):
    if not data_dir.exists():
        return []
    return [p for p in data_dir.iterdir() if p.is_file()]


def detect_env_csv_files(data_dir: Path):
    """
    환경 CSV: 파일명 하드코딩 없이,
    - 확장자 .csv
    - 파일명에 '환경' + '데이터' 포함
    - 학교명 포함
    으로 탐색
    """
    mapping = {}

    for p in list_files(data_dir):
        if p.suffix.lower() != ".csv":
            continue
        name = p.name
        name_nfc = norm_nfc(name)
        name_nfd = norm_nfd(name)

        # '환경'/'데이터' 포함 여부 (NFC/NFD 양방향)
        has_env = ("환경" in name_nfc and "데이터" in name_nfc) or ("환경" in name_nfd and "데이터" in name_nfd)
        if not has_env:
            continue

        for school in SCHOOLS:
            # 학교명 포함 여부도 양방향으로
            if (school in name_nfc) or (school in name_nfd):
                mapping[school] = p
                break

    return mapping


def detect_growth_xlsx_file(data_dir: Path):
    """
    생육 XLSX: 파일명 하드코딩 없이,
    - 확장자 .xlsx
    - 파일명에 '생육' 또는 '결과' 포함 (가능한 범위로)
    를 우선 탐지, 없으면 xlsx 하나라도 있으면 그걸 사용.
    """
    candidates = []
    for p in list_files(data_dir):
        if p.suffix.lower() != ".xlsx":
            continue
        name_nfc = norm_nfc(p.name)
        name_nfd = norm_nfd(p.name)
        score = 0
        if "생육" in name_nfc or "생육" in name_nfd:
            score += 2
        if "결과" in name_nfc or "결과" in name_nfd:
            score += 1
        candidates.append((score, p))

    if not candidates:
        return None

    # 점수 높은 순, 동점이면 이름순(안정성)
    candidates.sort(key=lambda x: (-x[0], norm_nfc(x[1].name)))
    return candidates[0][1]


def detect_school_from_sheet(sheet_name: str):
    """
    시트명 하드코딩 금지: sheet_name을 읽어 학교명을 '포함'으로 판별.
    """
    s_nfc = norm_nfc(sheet_name)
    s_nfd = norm_nfd(sheet_name)
    for school in SCHOOLS:
        if school in s_nfc or school in s_nfd:
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
            # 표준 컬럼 기대: time, temperature, humidity, ph, ec
            # time 파싱
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            # numeric 보정
            for col in ["temperature", "humidity", "ph", "ec"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            env_by_school[school] = df
        except Exception as e:
            errors[school] = f"{fp.name}: {e}"

    return env_by_school, errors, csv_map


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir_str: str):
    data_dir = Path(data_dir_str)
    xlsx_fp = detect_growth_xlsx_file(data_dir)
    if xlsx_fp is None:
        return {}, "생육 결과 XLSX 파일을 찾지 못했습니다.", None

    try:
        # 시트명 하드코딩 금지: sheet_name=None으로 전체 시트 로드
        sheet_dict = pd.read_excel(xlsx_fp, sheet_name=None, engine="openpyxl")
    except Exception as e:
        return {}, f"XLSX 로딩 실패: {e}", xlsx_fp

    growth_by_school = {}
    unknown_sheets = {}

    for sheet_name, df in sheet_dict.items():
        school = detect_school_from_sheet(sheet_name)
        # numeric 보정
        for col in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if school is None:
            unknown_sheets[sheet_name] = df
        else:
            growth_by_school[school] = df

    # unknown sheet가 있으면, 사용자에게 알려주되 앱은 진행
    warn_msg = None
    if unknown_sheets:
        warn_msg = "학교명을 포함하지 않는 시트가 있어(자동 매핑 불가) 제외했습니다: " + ", ".join(list(unknown_sheets.keys())[:10])

    return growth_by_school, warn_msg, xlsx_fp


def combine_env(env_by_school: dict, selected: str):
    if selected == "전체":
        frames = []
        for school, df in env_by_school.items():
            tmp = df.copy()
            tmp["school"] = school
            frames.append(tmp)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
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
    else:
        df = growth_by_school.get(selected, pd.DataFrame()).copy()
        if not df.empty:
            df["school"] = selected
            df["target_ec"] = TARGET_EC.get(selected)
        return df


# =========================
# Sidebar
# =========================
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)


# =========================
# Load data with spinner + errors
# =========================
DATA_DIR = Path(__file__).parent / "data"

with st.spinner("데이터를 불러오는 중..."):
    env_by_school, env_errors, env_csv_map = load_env_data(str(DATA_DIR))
    growth_by_school, growth_warn, growth_xlsx_fp = load_growth_data(str(DATA_DIR))

if not DATA_DIR.exists():
    st.error("data/ 폴더를 찾지 못했습니다. 프로젝트 구조를 확인하세요.")
    st.stop()

if env_errors:
    st.error("일부 환경 CSV 로딩 중 오류가 발생했습니다:\n- " + "\n- ".join([f"{k}: {v}" for k, v in env_errors.items()]))

if growth_warn:
    st.warning(growth_warn)

if not env_by_school:
    st.error("환경 데이터(CSV)를 찾지 못했습니다. data/ 폴더에 '환경데이터' CSV가 있는지 확인하세요.")
if not growth_by_school:
    st.error("생육 결과 데이터(XLSX)를 찾지 못했습니다. data/ 폴더에 생육 결과 XLSX가 있는지 확인하세요.")

if (not env_by_school) and (not growth_by_school):
    st.stop()


# =========================
# Prepare filtered / combined
# =========================
env_df = combine_env(env_by_school, selected_school)
growth_df = combine_growth(growth_by_school, selected_school)

# compute environment school means (for bar charts, always show all schools if possible)
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

# growth summaries
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
# Tab 1: Overview
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
    # 표: 학교명, EC 목표, 개체수, 색상
    condition_rows = []
    for school in SCHOOLS:
        n = int(growth_summ_df.loc[growth_summ_df["school"] == school, "n"].iloc[0]) if (not growth_summ_df.empty and (growth_summ_df["school"] == school).any()) else None
        condition_rows.append(
            {
                "학교명": school,
                "EC 목표": TARGET_EC.get(school),
                "개체수": n,
                "색상": SCHOOL_COLOR.get(school),
            }
        )
    cond_df = pd.DataFrame(condition_rows)
    st.dataframe(cond_df, use_container_width=True, hide_index=True)

    # KPI cards
    total_n = int(growth_summ_df["n"].sum()) if (not growth_summ_df.empty and "n" in growth_summ_df.columns) else 0

    # 전체 평균(데이터 존재하는 학교/행 기반)
    avg_temp = None
    avg_hum = None
    if not env_df.empty:
        avg_temp = env_df["temperature"].mean() if "temperature" in env_df.columns else None
        avg_hum = env_df["humidity"].mean() if "humidity" in env_df.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n:,}")
    c2.metric("평균 온도", "-" if avg_temp is None else f"{avg_temp:.2f} ℃")
    c3.metric("평균 습도", "-" if avg_hum is None else f"{avg_hum:.2f} %")
    c4.metric("최적 EC", "2.0 (하늘고)")

    st.info("팁: 사이드바에서 학교를 선택하면, 탭 2/3에서 해당 학교 시계열·분포를 바로 확인할 수 있어요.")


# -------------------------
# Tab 2: Environment
# -------------------------
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_means_df.empty:
        st.error("환경 평균 비교를 위한 데이터가 없습니다(CSV 로딩 실패 또는 컬럼 누락).")
    else:
        # 2x2 subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC"),
            horizontal_spacing=0.10,
            vertical_spacing=0.15,
        )

        # 평균 온도
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["temperature_mean"],
                name="평균 온도",
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in env_means_df["school"]]),
            ),
            row=1,
            col=1,
        )

        # 평균 습도
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["humidity_mean"],
                name="평균 습도",
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in env_means_df["school"]]),
            ),
            row=1,
            col=2,
        )

        # 평균 pH
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["ph_mean"],
                name="평균 pH",
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in env_means_df["school"]]),
            ),
            row=2,
            col=1,
        )

        # 목표 EC vs 실측 EC (이중 막대)
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["target_ec"],
                name="목표 EC",
                marker=dict(opacity=0.75),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=env_means_df["school"],
                y=env_means_df["ec_mean"],
                name="실측 평균 EC",
                marker=dict(opacity=0.75),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            barmode="group",
            height=650,
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(family=PLOTLY_FONT_FAMILY),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    if selected_school == "전체":
        st.info("시계열은 학교별로 측정 주기가 달라서, 사이드바에서 특정 학교를 선택하면 더 명확하게 볼 수 있어요.")
    else:
        if env_df.empty:
            st.error("선택한 학교의 환경 데이터가 없습니다.")
        else:
            # ensure sorted by time if exists
            if "time" in env_df.columns:
                env_df = env_df.sort_values("time")

            # 온도
            if "temperature" in env_df.columns:
                fig_t = px.line(env_df, x="time" if "time" in env_df.columns else env_df.index, y="temperature", title="온도 변화")
                fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.warning("temperature 컬럼이 없어 온도 시계열을 그릴 수 없습니다.")

            # 습도
            if "humidity" in env_df.columns:
                fig_h = px.line(env_df, x="time" if "time" in env_df.columns else env_df.index, y="humidity", title="습도 변화")
                fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.warning("humidity 컬럼이 없어 습도 시계열을 그릴 수 없습니다.")

            # EC (목표선)
            if "ec" in env_df.columns:
                fig_ec = px.line(env_df, x="time" if "time" in env_df.columns else env_df.index, y="ec", title="EC 변화 (목표 EC 수평선 포함)")
                target = TARGET_EC.get(selected_school)
                if target is not None:
                    fig_ec.add_hline(y=target, line_dash="dash", annotation_text=f"목표 EC={target}", annotation_position="top left")
                fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_ec, use_container_width=True)
            else:
                st.warning("ec 컬럼이 없어 EC 시계열을 그릴 수 없습니다.")

    st.divider()
    with st.expander("환경 데이터 원본 테이블 + CSV 다운로드"):
        if env_df.empty:
            st.error("표시할 환경 데이터가 없습니다.")
        else:
            st.dataframe(env_df, use_container_width=True)

            # CSV download (Bytes)
            csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ 환경 데이터 CSV 다운로드",
                data=csv_bytes,
                file_name="환경데이터_선택학교.csv" if selected_school != "전체" else "환경데이터_전체.csv",
                mime="text/csv",
            )


# -------------------------
# Tab 3: Growth results
# -------------------------
with tab3:
    st.subheader("핵심 결과")

    if growth_summ_df.empty or "mean_weight" not in growth_summ_df.columns:
        st.error("생육 결과 요약을 만들 수 없습니다(XLSX 로딩 실패 또는 컬럼 누락).")
    else:
        # max mean weight EC
        tmp = growth_summ_df.dropna(subset=["mean_weight"]).copy()
        best_row = tmp.sort_values("mean_weight", ascending=False).head(1)
        best_ec = None
        best_mean_weight = None
        best_school = None
        if not best_row.empty:
            best_ec = float(best_row["target_ec"].iloc[0])
            best_mean_weight = float(best_row["mean_weight"].iloc[0])
            best_school = str(best_row["school"].iloc[0])

        # 하늘고(EC 2.0) 강조 카드도 별도로
        sky_mean_weight = None
        if (growth_summ_df["school"] == "하늘고").any():
            sky_mean_weight = growth_summ_df.loc[growth_summ_df["school"] == "하늘고", "mean_weight"].iloc[0]

        c1, c2 = st.columns(2)
        if best_ec is None:
            c1.metric("EC별 평균 생중량(최대)", "-")
        else:
            c1.metric("EC별 평균 생중량(최대)", f"{best_mean_weight:.3f} g", help=f"최대 평균 생중량: {best_school} (EC {best_ec})")

        c2.metric(
            "하늘고(EC 2.0) 평균 생중량",
            "-" if (sky_mean_weight is None or pd.isna(sky_mean_weight)) else f"{float(sky_mean_weight):.3f} g",
            help="요구사항: 하늘고(EC 2.0)를 최적 조건으로 강조",
        )

        if best_ec is not None and abs(best_ec - 2.0) > 1e-9:
            st.info(f"데이터상 평균 생중량 최댓값은 EC {best_ec}에서 관측되었지만, 연구 설정상 **최적 조건(하늘고 EC 2.0)**도 함께 강조해 해석합니다.")

    st.divider()

    st.subheader("EC별 생육 비교 (2x2)")
    if growth_summ_df.empty:
        st.error("생육 비교 그래프를 그릴 데이터가 없습니다.")
    else:
        # Ensure order by target_ec
        gplot = growth_summ_df.dropna(subset=["target_ec"]).copy()
        gplot = gplot.sort_values("target_ec")

        fig2 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수", "평균 지상부 길이(mm)", "개체수 비교"),
            horizontal_spacing=0.10,
            vertical_spacing=0.15,
        )

        # 평균 생중량
        fig2.add_trace(
            go.Bar(
                x=gplot["target_ec"],
                y=gplot["mean_weight"],
                name="평균 생중량",
                text=gplot["school"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]]),
            ),
            row=1,
            col=1,
        )

        # 평균 잎 수
        fig2.add_trace(
            go.Bar(
                x=gplot["target_ec"],
                y=gplot["mean_leaf"],
                name="평균 잎 수",
                text=gplot["school"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]]),
            ),
            row=1,
            col=2,
        )

        # 평균 지상부 길이
        fig2.add_trace(
            go.Bar(
                x=gplot["target_ec"],
                y=gplot["mean_shoot"],
                name="평균 지상부 길이",
                text=gplot["school"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]]),
            ),
            row=2,
            col=1,
        )

        # 개체수
        fig2.add_trace(
            go.Bar(
                x=gplot["target_ec"],
                y=gplot["n"],
                name="개체수",
                text=gplot["school"],
                marker=dict(color=[SCHOOL_COLOR.get(s, "#888") for s in gplot["school"]]),
            ),
            row=2,
            col=2,
        )

        # 하늘고(EC 2.0) 수직선 강조 (가능하면)
        fig2.add_vline(x=2.0, line_dash="dash", annotation_text="하늘고 EC=2.0", annotation_position="top")

        fig2.update_layout(
            height=650,
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(family=PLOTLY_FONT_FAMILY),
            showlegend=False,
        )
        fig2.update_xaxes(title_text="목표 EC", row=1, col=1)
        fig2.update_xaxes(title_text="목표 EC", row=1, col=2)
        fig2.update_xaxes(title_text="목표 EC", row=2, col=1)
        fig2.update_xaxes(title_text="목표 EC", row=2, col=2)

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

    if growth_df.empty or ("생중량(g)" not in growth_df.columns):
        st.error("생중량 분포를 표시할 데이터가 없습니다.")
    else:
        # box or violin (요구사항: 박스플롯 또는 바이올린)
        fig_dist = px.violin(
            growth_df.dropna(subset=["생중량(g)"]),
            x="school",
            y="생중량(g)",
            box=True,
            points="all",
            title="학교별 생중량 분포 (바이올린 + 박스)",
            category_orders={"school": SCHOOLS},
        )
        fig_dist.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.subheader("상관관계 분석")

    if growth_df.empty:
        st.error("상관관계 분석을 위한 데이터가 없습니다.")
    else:
        colA, colB = st.columns(2)

        # 잎 수 vs 생중량
        with colA:
            if ("잎 수(장)" in growth_df.columns) and ("생중량(g)" in growth_df.columns):
                df_sc = growth_df.dropna(subset=["잎 수(장)", "생중량(g)"]).copy()
                fig_sc1 = px.scatter(
                    df_sc,
                    x="잎 수(장)",
                    y="생중량(g)",
                    color="school",
                    title="잎 수 vs 생중량",
                    color_discrete_map=SCHOOL_COLOR,
                )
                fig_sc1.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_sc1, use_container_width=True)
            else:
                st.warning("필요 컬럼(잎 수(장), 생중량(g))이 없어 산점도를 그릴 수 없습니다.")

        # 지상부 길이 vs 생중량
        with colB:
            if ("지상부 길이(mm)" in growth_df.columns) and ("생중량(g)" in growth_df.columns):
                df_sc2 = growth_df.dropna(subset=["지상부 길이(mm)", "생중량(g)"]).copy()
                fig_sc2 = px.scatter(
                    df_sc2,
                    x="지상부 길이(mm)",
                    y="생중량(g)",
                    color="school",
                    title="지상부 길이 vs 생중량",
                    color_discrete_map=SCHOOL_COLOR,
                )
                fig_sc2.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
                st.plotly_chart(fig_sc2, use_container_width=True)
            else:
                st.warning("필요 컬럼(지상부 길이(mm), 생중량(g))이 없어 산점도를 그릴 수 없습니다.")

    st.divider()
    with st.expander("학교별 생육 데이터 원본 + XLSX 다운로드"):
        if growth_df.empty:
            st.error("표시할 생육 데이터가 없습니다.")
        else:
            st.dataframe(growth_df, use_container_width=True)

        # XLSX 다운로드: 반드시 BytesIO + to_excel(buffer, ...)
        if growth_by_school:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                if selected_school == "전체":
                    # 로드된 학교만 기록 (시트명은 학교명으로 생성)
                    for school, df in growth_by_school.items():
                        df.to_excel(writer, sheet_name=school, index=False)
                else:
                    df_one = growth_by_school.get(selected_school, pd.DataFrame())
                    df_one.to_excel(writer, sheet_name=selected_school, index=False)

            buffer.seek(0)
            st.download_button(
                label="⬇️ 생육 데이터 XLSX 다운로드",
                data=buffer,
                file_name="생육데이터_선택학교.xlsx" if selected_school != "전체" else "생육데이터_전체.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # 원본 파일도 안내 (있으면)
        if growth_xlsx_fp is not None and growth_xlsx_fp.exists():
            st.caption(f"원본 파일 감지: {growth_xlsx_fp.name} (앱 내부에서 재구성 XLSX로 다운로드 제공)")


# Footer
st.caption("© Polar Plant EC Dashboard • Streamlit + Plotly")


