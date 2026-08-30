"""
visualize.py — FAISS 기반 RAG 청크 시각화 앱

실행:
  streamlit run visualize.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from rag import is_index_ready, load_chunks_df

# ─────────────────────────────────────────────
st.set_page_config(page_title="RAG 청크 시각화", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
html, body, [class*="css"] { font-family: 'Pretendard Variable', Pretendard, sans-serif; }
.metric-label { font-size: 0.8rem; color: #64748b; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #0891b2; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 RAG 청크 시각화")
st.caption("FAISS 인덱스에 저장된 청크 데이터를 탐색합니다.")

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
if not is_index_ready():
    st.warning("`faiss.index` / `chunks.pkl` 파일이 없습니다. `main.py`에서 인덱스를 먼저 빌드하세요.")
    st.stop()

@st.cache_data
def load_data():
    chunks, vecs = load_chunks_df()
    df = pd.DataFrame({
        "source": [c["source"] for c in chunks],
        "chunk_index": [c["chunk_index"] for c in chunks],
        "text": [c["text"] for c in chunks],
        "length": [len(c["text"]) for c in chunks],
    })
    return df, vecs

with st.spinner("데이터 불러오는 중…"):
    df, embeddings = load_data()

if df.empty:
    st.warning("인덱스가 비어 있습니다.")
    st.stop()

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab_overview, tab_browser, tab_embed = st.tabs(["📊 개요", "📄 청크 탐색", "🗺️ 임베딩 공간"])

# ── 탭 1: 개요 ──────────────────────────────
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("총 청크 수", f"{len(df):,}"),
        ("PDF 파일 수", f"{df['source'].nunique()}"),
        ("평균 청크 길이", f"{int(df['length'].mean()):,}자"),
        ("최대 청크 길이", f"{df['length'].max():,}자"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("파일별 청크 수")
        sc = df.groupby("source").size().reset_index(name="청크 수").sort_values("청크 수")
        fig = px.bar(sc, x="청크 수", y="source", orientation="h",
                     color="청크 수", color_continuous_scale="Blues")
        fig.update_layout(height=max(250, len(sc) * 40 + 80), coloraxis_showscale=False,
                          margin=dict(l=0, r=10, t=10, b=30),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("청크 길이 분포")
        fig = px.histogram(df, x="length", nbins=40, color_discrete_sequence=["#0891b2"],
                           labels={"length": "글자 수"})
        fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=30), bargap=0.05,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(gridcolor="#e2e8f0")
        fig.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("파일별 청크 길이 분포 (Box)")
    fig = px.box(df, x="source", y="length", color="source",
                 labels={"source": "파일", "length": "글자 수"}, points="outliers")
    fig.update_layout(height=350, showlegend=False, margin=dict(l=0, r=10, t=10, b=60),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

# ── 탭 2: 청크 탐색 ─────────────────────────
with tab_browser:
    st.subheader("청크 탐색")
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        sources = ["전체"] + sorted(df["source"].unique().tolist())
        sel_source = st.selectbox("파일 필터", sources)
    with f2:
        keyword = st.text_input("키워드 검색", placeholder="찾을 내용 입력…")
    with f3:
        lo, hi = int(df["length"].min()), int(df["length"].max())
        len_range = st.slider("길이 범위", lo, hi, (lo, hi))

    filtered = df.copy()
    if sel_source != "전체":
        filtered = filtered[filtered["source"] == sel_source]
    if keyword:
        filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]
    filtered = filtered[filtered["length"].between(*len_range)]

    st.caption(f"총 {len(df):,}개 중 **{len(filtered):,}개** 표시")

    for _, row in filtered.head(50).iterrows():
        with st.expander(f"[{row['source']}] 청크 #{row['chunk_index']}  —  {row['length']}자"):
            text = row["text"]
            if keyword:
                text = re.sub(f"({re.escape(keyword)})", r"**\1**", text, flags=re.IGNORECASE)
            st.markdown(text)

    if len(filtered) > 50:
        st.info("처음 50개만 표시됩니다. 키워드나 필터로 범위를 좁혀 보세요.")

# ── 탭 3: 임베딩 공간 ───────────────────────
with tab_embed:
    st.subheader("임베딩 2D 시각화 (PCA)")
    st.caption("384차원 벡터를 PCA로 2차원 축소해 청크 분포를 시각화합니다.")

    with st.spinner("PCA 계산 중…"):
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)

    df_pca = df[["source", "chunk_index", "length"]].copy()
    df_pca["PC1"] = coords[:, 0]
    df_pca["PC2"] = coords[:, 1]
    df_pca["preview"] = df["text"].str[:80] + "…"
    explained = pca.explained_variance_ratio_ * 100

    fig = px.scatter(
        df_pca, x="PC1", y="PC2", color="source",
        size="length", size_max=18, opacity=0.75,
        hover_data={"PC1": False, "PC2": False,
                    "chunk_index": True, "length": True, "preview": True},
        labels={"PC1": f"PC1 ({explained[0]:.1f}%)", "PC2": f"PC2 ({explained[1]:.1f}%)"},
    )
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=10, b=30),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"누적 분산 설명률: {sum(explained):.1f}%  (PC1: {explained[0]:.1f}%, PC2: {explained[1]:.1f}%)")
