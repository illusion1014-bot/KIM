import os
import streamlit as st

# Streamlit 설정을 먼저 실행
st.set_page_config(
    page_title="노인 생활 속 건강 관리 챗봇",
    page_icon="🏥",
    layout="centered",
)

import anthropic
from dotenv import load_dotenv

try:
    from rag import build_index, query_context, is_index_ready, query_context_with_web
except Exception as e:
    st.error(f"RAG 모듈 로드 오류: {e}")
    st.stop()

# 로컬: .env 로드 / Streamlit Cloud: st.secrets 사용
load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """st.secrets(Streamlit Cloud) → os.environ(.env) → default 순으로 조회."""
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

# ────────────────────────────────────────────────────────────────────────────
# 노인 생활 속 건강 관리 챗봇 프로필 (시스템 프롬프트에 항상 포함)
# ────────────────────────────────────────────────────────────────────────────
PROFILE = """
당신은 노인의 건강한 생활을 돕기 위한 건강 관리 어시스턴트입니다.

[챗봇 정보]
이름: 노인 생활 속 건강 관리 챗봇
목적: 노인의 일상 건강 관리 지원 및 웰빙 정보 제공
전문 분야: 영양 관리, 운동/재활, 만성질환 관리, 건강한 생활 습관

[제공 서비스]
- 노인 맞춤 영양 가이드 및 식이 정보
- 안전한 운동 및 재활 프로그램 정보
- 만성질환(당뇨, 고혈압, 관절염 등) 관리 조언
- 일상 건강 습관 및 예방 정보

[답변 방침]
- 노인 건강 관리에 관한 질문: 과학 기반 정보로 답변
- PDF 자료가 제공된 경우: 해당 자료를 우선 참고하여 답변하고, 출처를 명시
- 의료 조언이 필요한 경우: "반드시 의사와 상담하시기 바랍니다" 명시
- 긴급 상황: "증상이 심하면 즉시 119에 신고하세요" 안내
- 모르는 내용: "의료 자료에서 확인할 수 없는 내용입니다. 전문가와 상담해주세요"라고 답변
- 모든 답변은 한국어로, 존댓글 사용하여 친근하고 존중하는 톤 유지
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Streamlit 설정 (위에서 이미 실행됨)
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
html, body, [class*="css"] { font-family: 'Pretendard Variable', Pretendard, sans-serif; }

/* 메인 배경 및 테마 */
[data-testid="stAppViewContainer"] { background-color: #F2EDD5; }

/* 소스 박스 스타일 (의료 정보 출처) */
.source-box {
    background: rgba(8, 74, 36, 0.08);
    border-left: 4px solid #084A24;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #04261E;
    border-radius: 0 6px 6px 0;
    margin-top: 0.8rem;
    font-weight: 500;
}

/* 제목 색상 */
h1 { color: #084A24 !important; }
h2, h3 { color: #084A24 !important; }

/* 버튼 색상 */
.stButton>button {
    background-color: #084A24 !important;
    color: white !important;
    border: none !important;
}

.stButton>button:hover {
    background-color: #F26716 !important;
}

/* 긴급 경고 메시지 */
.emergency-warning {
    background-color: #E7390D;
    color: white;
    padding: 1rem;
    border-radius: 8px;
    font-weight: bold;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 노인 생활 속 건강 관리 챗봇")
st.caption("노인의 건강한 생활을 위한 영양, 운동, 질환 관리 정보를 제공합니다. 건강 관리 관련 어떤 질문이든 물어보세요!")

# ────────────────────────────────────────────────────────────────────────────
# 사이드바 — API 키 & RAG 인덱스 관리
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API 키",
            type="password",
            placeholder="sk-ant-...",
        )

    st.divider()
    st.subheader("📚 PDF 인덱스")

    ready = is_index_ready()
    if ready:
        st.success("인덱스 준비 완료")
    else:
        st.warning("인덱스 없음 — 관리자 로그인 후 빌드하세요.")

    # 관리자 잠금 영역
    if "admin_unlocked" not in st.session_state:
        st.session_state.admin_unlocked = False

    if not st.session_state.admin_unlocked:
        admin_pw = st.text_input("관리자 비밀번호", type="password", placeholder="비밀번호 입력…")
        if st.button("로그인", use_container_width=True):
            correct = get_secret("ADMIN_PASSWORD", "admin1234")
            if admin_pw == correct:
                st.session_state.admin_unlocked = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    else:
        st.success("관리자 모드")
        if st.button("🔄 인덱스 빌드 / 재빌드", use_container_width=True):
            with st.spinner("PDF 파싱 & 임베딩 중… (첫 실행 시 수 분 소요)"):
                try:
                    count = build_index()
                    st.success(f"완료: {count}개 청크 저장")
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
        if st.button("잠금", use_container_width=True, type="secondary"):
            st.session_state.admin_unlocked = False
            st.rerun()

    use_rag = st.toggle("PDF 자료 검색", value=ready, disabled=not ready)
    use_web = st.toggle("인터넷 검색 (Wikipedia + DuckDuckGo)", value=True)

    st.divider()
    n_results = st.slider("검색할 청크 수", 1, 10, 5)

if api_key:
    client = anthropic.Anthropic(api_key=api_key)
else:
    client = None

# ────────────────────────────────────────────────────────────────────────────
# 대화
# ────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요…"):
    if not client:
        st.error("❌ API 키가 설정되지 않았습니다. 사이드바에서 Anthropic API 키를 입력하세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 통합 검색 (PDF + 웹)
        rag_context = ""
        if use_rag and ready:
            rag_context = query_context_with_web(prompt, n_results=n_results, use_web=use_web)
        elif use_web:
            # PDF 없으면 웹 검색만
            rag_context = query_context_with_web(prompt, n_results=0, use_web=True)

        # 시스템 프롬프트 구성
        system_prompt = PROFILE
        if rag_context:
            system_prompt += f"\n\n[PDF 자료에서 검색된 관련 내용]\n{rag_context}"

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중…"):
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                )
                answer = response.content[0].text

            st.markdown(answer)

            # 참고 자료 표시
            if rag_context:
                # 출처별 분류
                sources_pdf = set()
                sources_wiki = set()
                sources_web = set()

                for line in rag_context.splitlines():
                    if "[출처:" in line:
                        if "Wikipedia" in line:
                            src = line.split(" - ")[0].replace("[출처: Wikipedia", "").strip()
                            sources_wiki.add(src)
                        elif line.startswith("[출처:") and "|" in line:
                            src = line.split("|")[0].replace("[출처:", "").strip()
                            sources_pdf.add(src)
                        else:
                            src = line.split(" - ")[0].replace("[출처:", "").strip()
                            sources_web.add(src)

                sources_text = []
                if sources_pdf:
                    sources_text.append(f"📚 PDF: {', '.join(sorted(sources_pdf))}")
                if sources_wiki:
                    sources_text.append(f"📖 Wikipedia: {', '.join(sorted(sources_wiki))}")
                if sources_web:
                    sources_text.append(f"🌐 웹: {', '.join(sorted(sources_web))}")

                if sources_text:
                    st.markdown(
                        "<div class='source-box'>📌 참고 자료:<br>" +
                        "<br>".join(sources_text) +
                        "</div>",
                        unsafe_allow_html=True,
                    )

            st.session_state.messages.append({"role": "assistant", "content": answer})

# 초기화 버튼
if st.session_state.messages:
    if st.button("대화 초기화", type="secondary"):
        st.session_state.messages = []
        st.rerun()
