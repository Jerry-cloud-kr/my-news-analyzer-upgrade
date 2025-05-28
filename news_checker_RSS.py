import streamlit as st
from newspaper import Article
# from sentence_transformers import SentenceTransformer, util # <<<<<<<<<<< 일단 주석 처리
import openai # OpenAI 라이브러리
from openai import OpenAI # OpenAI 클라이언트 클래스 임포트
import google.generativeai as genai
import feedparser # 키워드 검색 기능에 필요

# --- OpenAI API Key 및 클라이언트 설정 (Secrets 사용) ---
client_openai = None # OpenAI 클라이언트 변수 선언
OPENAI_API_KEY_Direct_Placeholder = "YOUR_OPENAI_KEY_PLACEHOLDER" # 로컬 테스트용 플레이스홀더

try:
    OPENAI_API_KEY_FROM_SECRETS = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY_FROM_SECRETS:
         st.error("⚠️ OpenAI API 키가 Streamlit Secrets에 설정되었으나 값이 비어있습니다. 앱 설정에서 확인해주세요.")
         st.stop()
    client_openai = OpenAI(api_key=OPENAI_API_KEY_FROM_SECRETS) 
except KeyError:
    if OPENAI_API_KEY_Direct_Placeholder == "YOUR_OPENAI_KEY_PLACEHOLDER" or not OPENAI_API_KEY_Direct_Placeholder:
        st.error("OpenAI API 키를 Streamlit Secrets에서 찾을 수 없습니다. 로컬 테스트를 원하시면 코드 상단의 OPENAI_API_KEY_Direct_Placeholder 값을 실제 키로 입력하거나, 앱 배포 후 Streamlit Community Cloud의 Secrets 설정을 확인하세요.")
        st.stop()
    else: 
        st.warning("로컬 테스트용 OpenAI API 키가 코드에 직접 설정되어 있습니다. GitHub에 배포/푸시하기 전에 이 부분을 반드시 Streamlit Secrets 방식으로 변경하거나 키를 삭제하세요.", icon="❗")
        client_openai = OpenAI(api_key=OPENAI_API_KEY_Direct_Placeholder)
except Exception as e:
    st.error(f"OpenAI API 키 설정 또는 클라이언트 초기화 중 오류: {e}")
    st.stop()

if client_openai is None: 
    st.error("OpenAI 클라이언트가 초기화되지 않았습니다. API 키 설정을 확인해주세요.")
    st.stop()

# --- Google AI API Key 설정 (Secrets 사용) ---
GOOGLE_AI_API_KEY_Direct_Placeholder = "YOUR_GOOGLE_AI_KEY_PLACEHOLDER" 
try:
    GOOGLE_AI_API_KEY_FROM_SECRETS = st.secrets["GOOGLE_AI_API_KEY"]
    if not GOOGLE_AI_API_KEY_FROM_SECRETS:
         st.error("⚠️ Google AI API 키가 Streamlit Secrets에 설정되었으나 값이 비어있습니다. 앱 설정에서 확인해주세요.")
         st.stop()
    genai.configure(api_key=GOOGLE_AI_API_KEY_FROM_SECRETS)
except KeyError:
    if GOOGLE_AI_API_KEY_Direct_Placeholder == "YOUR_GOOGLE_AI_KEY_PLACEHOLDER" or not GOOGLE_AI_API_KEY_Direct_Placeholder:
        st.error("Google AI API 키를 Streamlit Secrets에서 찾을 수 없습니다. 로컬 테스트를 원하시면 코드 상단의 GOOGLE_AI_API_KEY_Direct_Placeholder 값을 실제 키로 입력하거나, 앱 배포 후 Streamlit Community Cloud의 Secrets 설정을 확인하세요.")
        st.stop()
    else: 
        st.warning("로컬 테스트용 Google AI API 키가 코드에 직접 설정되어 있습니다. GitHub에 배포/푸시하기 전에 이 부분을 반드시 Streamlit Secrets 방식으로 변경하거나 키를 삭제하세요.", icon="❗")
        genai.configure(api_key=GOOGLE_AI_API_KEY_Direct_Placeholder)
except Exception as e:
    st.error(f"Google AI API 키 설정 중 오류: {e}")
    st.stop()


# 요약 함수 (Gemini 사용)
def summarize_text_gemini(text_content): # 함수 이름에 _gemini 명시
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction="너는 뉴스 기사의 핵심 내용을 객관적으로 요약하는 AI야."
    )
    prompt = f"""
    다음 뉴스 기사 본문을 객관적인 사실에 기반하여 핵심 내용 중심으로 요약해 주십시오.
    요약에는 주요 인물, 발생한 사건, 중요한 발언, 그리고 사건의 배경 정보가 포함되어야 합니다.
    주관적인 해석, 평가, 또는 기사에 명시적으로 드러나지 않은 추론은 배제하고, 사실 관계를 명확히 전달하는 데 집중해 주십시오.
    분량은 한국어 기준으로 약 3~5문장 (또는 100~150 단어) 정도로 간결하게 작성해 주십시오.

    기사:
    {text_content}
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3)
        )
        return response.text.strip()
    except Exception as e:
        st.warning("요약 생성 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        print(f"Gemini 요약 API 오류: {e}")
        return "요약 생성에 실패했습니다."

# 프레이밍 분석 함수 (OpenAI GPT 사용 - 최신 SDK 적용)
def detect_bias_openai(title, text_content): # 함수 이름에 _openai 명시
    prompt = f"""
    다음은 뉴스 제목과 본문입니다.
    제목이 본문 내용을 충분히 반영하고 있는지, 중요한 맥락이나 인물의 입장이 왜곡되거나 누락되었는지 판단해줘.

    제목: {title}
    본문: {text_content}

    분석 결과를 간단히 3~5줄로 정리해줘.
    """
    try:
        completion = client_openai.chat.completions.create(
            model="gpt-4", # 또는 "gpt-4o" 
            messages=[
                {"role": "system", "content": "너는 공정한 뉴스 프레이밍 분석 도우미야."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning("프레이밍 분석 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        print(f"OpenAI 프레이밍 분석 API 오류: {e}")
        return "프레이밍 분석에 실패했습니다."


# Gemini 기반 키워드 추출 함수 (새로 변경된 부분)
def extract_keywords_gemini(article_text):
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction="You are an AI assistant specialized in extracting the most important keywords from news articles. Keywords should be nouns or core noun phrases. Respond only with the keywords, separated by commas."
    )
    user_prompt = f"""
    다음 뉴스 기사 본문에서 가장 중요한 핵심 키워드를 5개만 추출하여, 각 키워드를 쉼표(,)로 구분한 하나의 문자열로 응답해주세요. 다른 설명이나 문장은 포함하지 마세요.

    예시 응답:
    키워드1,핵심 단어,세번째 키워드,중요 개념,마지막

    기사 본문:
    {article_text}
    """
    try:
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        keywords_string = response.text.strip()
        if keywords_string:
            if "\n" in keywords_string: 
                keywords_string = keywords_string.split("\n")[-1]
            if ":" in keywords_string:
                keywords_string = keywords_string.split(":")[-1].strip()
            return [kw.strip() for kw in keywords_string.split(',') if kw.strip()]
        else:
            return []
    except Exception as e:
        print(f"Gemini 키워드 추출 API 오류: {e}")
        st.warning("AI 키워드 추출 중 일시적인 오류가 발생했습니다.")
        return []

# --- 유사도 측정 모델 로드 (일단 주석 처리) ---
# try:
#     model_similarity = SentenceTransformer('all-MiniLM-L6-v2')
# except Exception as e:
#     st.error(f"SentenceTransformer 모델 로드 중 오류: {e}. 모델이 올바르게 설치되었는지 확인해주세요.")
#     st.stop()

# --- 기사 분석 및 결과 표시 함수 ---
def display_article_analysis_content(title_to_display, text_content, article_url):
    st.markdown("---")
    st.subheader("📰 기사 제목")
    st.write(f"**{title_to_display}**")
    st.markdown(f"[🔗 기사 원문 바로가기]({article_url})", unsafe_allow_html=True)
    st.markdown("---")

    # Gemini로 요약
    st.subheader("🧾 본문 요약 (by Gemini AI)")
    with st.expander("⚠️ AI 요약에 대한 중요 안내 (클릭하여 확인)"):
        st.markdown("""
        **주의: AI 기반 요약 (Gemini)**

        * 본 요약은 Gemini 모델을 통해 생성되었으며, 기사의 모든 내용을 완벽하게 반영하지 못할 수 있습니다.
        * AI는 학습 데이터의 한계나 요약 과정의 특성으로 인해 때때로 부정확한 내용을 전달하거나 중요한 내용을 생략할 수 있습니다.
        * 제공된 요약은 기사의 핵심 내용을 빠르게 파악하기 위한 참고 자료로만 활용해주십시오.
        * 기사의 전체적인 맥락과 정확한 정보 확인을 위해서는 반드시 원문 기사를 함께 읽어보시는 것이 중요하며, 최종적인 내용에 대한 판단은 사용자의 책임입니다.
        """)
    body_summary = summarize_text_gemini(text_content) # Gemini 요약 함수 호출
    st.write(body_summary)
    st.markdown("---")

    # Gemini로 키워드 추출 및 비교 (변경된 부분)
    st.subheader("🔍 AI 추출 주요 키워드와 제목 비교 (by Gemini AI)")
    extracted_keywords = extract_keywords_gemini(text_content) # Gemini 키워드 추출 함수 호출
    if not extracted_keywords:
        st.info("ℹ️ AI가 본문에서 주요 키워드를 추출하지 못했거나, 추출된 키워드가 없습니다.")
    else:
        st.caption(f"AI(Gemini)가 본문에서 추출한 주요 키워드: **{', '.join(extracted_keywords)}**") # UI 텍스트 변경
        missing_in_title = [kw for kw in extracted_keywords if kw.lower() not in title_to_display.lower()]
        if missing_in_title:
            st.warning(f"❗ AI 추출 키워드 중 일부가 제목에 빠져있을 수 있습니다: **{', '.join(missing_in_title)}**")
        else:
            st.success("✅ AI 추출 핵심 키워드가 제목에 잘 반영되어 있습니다.")
    st.markdown("---")
    
    # 유사도 판단 (일단 주석 처리)
    st.subheader("📊 제목-본문요약 유사도 판단 (현재 비활성화)")
    st.info("ℹ️ 제목-본문 유사도 분석 기능은 현재 SentenceTransformer 모델 로드 오류로 인해 비활성화되어 있습니다.")
    st.markdown("---")

    # GPT로 프레이밍 분석 (유지)
    st.subheader("🕵️ 프레이밍 분석 결과 (by GPT)")
    with st.expander("⚠️ AI 프레이밍 분석 주의사항 (클릭하여 확인)"):
        st.markdown("""
        **주의: AI 기반 프레이밍 분석 (GPT)**

        * 본 분석은 GPT 모델에 의해 수행되었으며, 완벽성을 보장하지 않습니다.
        * AI는 데이터와 학습 방식에 따라 편향된 결과를 제시할 수도 있습니다.
        * 제공된 분석은 참고 자료로 활용하시고, 최종적인 판단은 사용자의 책임하에 이루어져야 합니다.
        """)
    framing_result = detect_bias_openai(title_to_display, text_content) # GPT 프레이밍 분석 함수 호출
    st.info(framing_result)


# --- Streamlit 앱 UI 구성 ---
st.set_page_config(page_title="뉴스읽은척방지기 (하이브리드)", page_icon="🧐")
st.title("🧐 뉴스읽은척방지기")
st.write("키워드 검색 또는 URL 직접 입력으로 뉴스 기사를 AI와 함께 분석해보세요!")
st.caption("본문 요약 및 키워드 추출은 Gemini AI, 프레이밍 분석은 OpenAI GPT를 사용합니다.") # 캡션 수정


tab1, tab2 = st.tabs(["🗂️ 키워드로 뉴스 검색/분석", "🔗 URL 직접 입력/분석"])

with tab1:
    st.subheader("키워드로 뉴스 찾아 분석하기")
    search_query_tab1 = st.text_input("검색할 키워드를 입력하세요:", placeholder="예: 애플 AI 전략", key="search_query_tab1")

    if st.button("🔍 뉴스 검색", key="search_button_tab1", use_container_width=True):
        if not search_query_tab1:
            st.warning("검색어를 입력해주세요.")
        else:
            st.session_state.search_results_tab1 = None 
            google_news_rss_url = f"https://news.google.com/rss/search?q={search_query_tab1}&hl=ko&gl=KR&ceid=KR:ko"
            try:
                with st.spinner(f"'{search_query_tab1}' 관련 뉴스를 검색 중..."):
                    feed = feedparser.parse(google_news_rss_url)
                if not feed.entries:
                    st.warning("검색 결과가 없습니다. 다른 검색어를 사용해보세요.")
                else:
                    st.success(f"'{search_query_tab1}' 관련 뉴스 {len(feed.entries)}건을 찾았습니다. (최대 30개 표시)")
                    st.session_state.search_results_tab1 = {entry.title: entry.link for entry in feed.entries[:30]}
            except Exception as e:
                st.error(f"뉴스 검색 중 오류 발생: {e}")
                st.session_state.search_results_tab1 = None
    
    if 'search_results_tab1' in st.session_state and st.session_state.search_results_tab1:
        selected_title_tab1 = st.selectbox(
            "분석할 기사를 선택하세요:",
            options=list(st.session_state.search_results_tab1.keys()),
            index=None, 
            placeholder="검색된 뉴스 목록에서 기사를 선택하세요...",
            key="searched_article_selectbox_tab1"
        )
        if selected_title_tab1 and st.button("👆 선택한 뉴스 분석하기", key="analyze_searched_button_tab1", use_container_width=True):
            selected_url_tab1 = st.session_state.search_results_tab1[selected_title_tab1]
            st.info(f"선택한 기사 분석 중: {selected_title_tab1}")
            try:
                with st.spinner(f"'{selected_title_tab1}' 기사를 가져와 AI가 분석 중입니다..."):
                    article = Article(selected_url_tab1, language='ko')
                    article.download()
                    article.parse()
                    if not article.title or not article.text or len(article.text) < 50:
                        st.error("선택한 기사의 제목이나 본문을 가져오지 못했거나 내용이 너무 짧습니다.")
                    else:
                        display_article_analysis_content(article.title, article.text, selected_url_tab1)
            except Exception as e:
                st.error(f"선택한 기사 처리 중 오류 발생: {e}")

with tab2:
    st.subheader("URL로 직접 뉴스 분석하기")
    url_direct_input_tab2 = st.text_input("분석할 뉴스 기사의 전체 URL을 입력해주세요:", placeholder="예: https://www.example-news.com/news/article123", key="url_direct_input_tab2")

    if st.button("🚀 URL 분석 시작", use_container_width=True, key="direct_url_analyze_button_tab2"):
        if not url_direct_input_tab2:
            st.warning("분석할 기사의 URL을 입력해주세요.")
        elif not (url_direct_input_tab2.startswith('http://') or url_direct_input_tab2.startswith('https://')):
            st.warning("올바른 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작해야 합니다.")
        else:
            st.info(f"입력하신 URL의 기사를 분석합니다: {url_direct_input_tab2}")
            try:
                with st.spinner(f"기사를 가져와 AI가 분석 중입니다..."):
                    article = Article(url_direct_input_tab2, language='ko')
                    article.download()
                    article.parse()
                    if not article.title or not article.text or len(article.text) < 50:
                        st.error("기사 제목이나 본문을 가져오지 못했거나 내용이 너무 짧습니다. 다른 URL을 시도해주세요.")
                    else:
                        display_article_analysis_content(article.title, article.text, url_direct_input_tab2)
            except Exception as e:
                st.error(f"URL 기사 처리 중 오류 발생: {e}")
                print(f"전체 오류: {e}") 
                st.caption("URL을 확인하시거나, 다른 기사를 시도해보세요. 일부 웹사이트는 외부 접근을 통한 기사 수집을 허용하지 않을 수 있습니다.")