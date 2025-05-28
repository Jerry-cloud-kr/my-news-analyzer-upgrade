import streamlit as st
import pandas as pd
import os
from newspaper import Article, Config # newspaper Config 추가
# from sentence_transformers import SentenceTransformer, util # <<<<<<<<<<< 일단 주석 처리
import openai # OpenAI 라이브러리
from openai import OpenAI # OpenAI 클라이언트 클래스 임포트
import google.generativeai as genai
import feedparser # 키워드 검색 기능에 필요
import requests # Naver 뉴스 원문 링크 추출에 필요
from bs4 import BeautifulSoup # Naver 뉴스 원문 링크 추출에 필요

# --- OpenAI API Key 및 클라이언트 설정 (Secrets 사용) ---
client_openai = None 
OPENAI_API_KEY_Direct_Placeholder = "YOUR_OPENAI_KEY_PLACEHOLDER" 

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

# --- Naver 뉴스 원문 링크 추출 함수 ---
@st.cache_data # 결과 캐싱 (동일 URL에 대해 반복 호출 방지)
def get_original_url_from_naver_news(naver_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(naver_url, headers=headers, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 일반적인 '기사원문' 링크 선택자 (Naver 구조 변경 시 업데이트 필요)
        original_link_tag = soup.select_one("a.media_end_head_origin_link_text")
        
        if original_link_tag and original_link_tag.get('href'):
            return original_link_tag['href']
        
        # 추가적인 선택자 시도 (예시)
        # press_logo = soup.select_one("div.press_logo img")
        # if press_logo and press_logo.parent.name == 'a' and press_logo.parent.get('href'):
        #     return press_logo.parent['href']
            
        print(f"Naver '기사원문' 링크 자동 추출 실패: {naver_url}")
        return None 
    except requests.exceptions.RequestException as e:
        print(f"Naver 뉴스 페이지 요청 중 오류 ({naver_url}): {e}")
        return None
    except Exception as e:
        print(f"Naver 뉴스 파싱 중 기타 오류 ({naver_url}): {e}")
        return None

# --- AI 기능 함수들 ---
def summarize_text_gemini(text_content):
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction="너는 뉴스 기사의 핵심 내용을 객관적으로 요약하는 AI야."
    )
    prompt = f"다음 뉴스 기사 본문을 객관적인 사실에 기반하여 핵심 내용 중심으로 요약해 주십시오. 요약에는 주요 인물, 발생한 사건, 중요한 발언, 그리고 사건의 배경 정보가 포함되어야 합니다. 주관적인 해석, 평가, 또는 기사에 명시적으로 드러나지 않은 추론은 배제하고, 사실 관계를 명확히 전달하는 데 집중해 주십시오. 분량은 한국어 기준으로 약 3~5문장 (또는 100~150 단어) 정도로 간결하게 작성해 주십시오.\n\n기사:\n{text_content}"
    try:
        response = model.generate_content(prompt,generation_config=genai.types.GenerationConfig(temperature=0.3))
        return response.text.strip()
    except Exception as e:
        st.warning("요약 생성 중 일시적인 오류가 발생했습니다.")
        print(f"Gemini 요약 API 오류: {e}")
        return "요약 생성에 실패했습니다."

def detect_bias_openai(title, text_content):
    prompt = f"다음은 뉴스 제목과 본문입니다. 제목이 본문 내용을 충분히 반영하고 있는지, 중요한 맥락이나 인물의 입장이 왜곡되거나 누락되었는지 판단해줘.\n\n제목: {title}\n본문: {text_content}\n\n분석 결과를 간단히 3~5줄로 정리해줘."
    try:
        completion = client_openai.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": "너는 공정한 뉴스 프레이밍 분석 도우미야."}, {"role": "user", "content": prompt}])
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning("프레이밍 분석 중 일시적인 오류가 발생했습니다.")
        print(f"OpenAI 프레이밍 분석 API 오류: {e}")
        return "프레이밍 분석에 실패했습니다."

def extract_keywords_gemini(article_text):
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction="You are an AI assistant specialized in extracting the most important keywords from news articles. Keywords should be nouns or core noun phrases. Respond only with the keywords, separated by commas."
    )
    user_prompt = f"다음 뉴스 기사 본문에서 가장 중요한 핵심 키워드를 5개만 추출하여, 각 키워드를 쉼표(,)로 구분한 하나의 문자열로 응답해주세요. 다른 설명이나 문장은 포함하지 마세요.\n\n예시 응답:\n키워드1,핵심 단어,세번째 키워드,중요 개념,마지막\n\n기사 본문:\n{article_text}"
    try:
        response = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
        keywords_string = response.text.strip()
        if keywords_string:
            if "\n" in keywords_string: keywords_string = keywords_string.split("\n")[-1]
            if ":" in keywords_string: keywords_string = keywords_string.split(":")[-1].strip()
            return [kw.strip() for kw in keywords_string.split(',') if kw.strip()]
        return []
    except Exception as e:
        print(f"Gemini 키워드 추출 API 오류: {e}")
        st.warning("AI 키워드 추출 중 일시적인 오류가 발생했습니다.")
        return []

# --- 유사도 측정 모델 로드 (일단 주석 처리) ---
# model_similarity = None 
# try:
#     model_similarity = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
#     if model_similarity is None: 
#         st.error("SentenceTransformer 모델 로드에 실패했으나 명시적 오류가 발생하지 않았습니다. 앱 실행을 중단합니다.")
#         st.stop()
# except Exception as e:
#     st.error(f"SentenceTransformer 모델 로드 중 오류: {e}")
#     st.error("팁: 이 오류는 보통 torch 또는 sentence-transformers 라이브러리 설치/호환성 문제입니다.")
#     st.info("유사도 분석 기능 없이 앱을 계속 사용하시려면 코드에서 해당 모델 로드 부분을 다시 주석 처리해주세요.")
#     st.stop()

# --- 기사 분석 및 결과 표시 함수 ---
def display_article_analysis_content(title_to_display, text_content, article_url):
    st.markdown("---")
    st.subheader("📰 기사 제목")
    st.write(f"**{title_to_display}**")
    st.markdown(f"[🔗 기사 원문 바로가기]({article_url})", unsafe_allow_html=True) # 여기서 article_url은 최종 분석 대상 URL
    st.markdown("---")

    # Gemini로 요약
    st.subheader("🧾 본문 요약 (by Gemini AI)")
    with st.expander("⚠️ AI 요약에 대한 중요 안내 (클릭하여 확인)", expanded=False):
        st.markdown(""" **주의: AI 기반 요약 (Gemini)**\n\n* 본 요약은 Gemini 모델을 통해 생성되었으며, 기사의 모든 내용을 완벽하게 반영하지 못할 수 있습니다.\n* AI는 학습 데이터의 한계나 요약 과정의 특성으로 인해 때때로 부정확한 내용을 전달하거나 중요한 내용을 생략할 수 있습니다.\n* 제공된 요약은 기사의 핵심 내용을 빠르게 파악하기 위한 참고 자료로만 활용해주십시오.\n* 기사의 전체적인 맥락과 정확한 정보 확인을 위해서는 반드시 원문 기사를 함께 읽어보시는 것이 중요하며, 최종적인 내용에 대한 판단은 사용자의 책임입니다. """)
    body_summary = summarize_text_gemini(text_content)
    st.write(body_summary)
    st.markdown("---")

    # Gemini로 키워드 추출 및 비교
    st.subheader("🔍 AI 추출 주요 키워드와 제목 비교 (by Gemini AI)")
    extracted_keywords = extract_keywords_gemini(text_content)
    if not extracted_keywords:
        st.info("ℹ️ AI가 본문에서 주요 키워드를 추출하지 못했거나, 추출된 키워드가 없습니다.")
    else:
        st.caption(f"AI(Gemini)가 본문에서 추출한 주요 키워드: **{', '.join(extracted_keywords)}**")
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
        st.markdown(""" **주의: AI 기반 프레이밍 분석 (GPT)**\n\n* 본 분석은 GPT 모델에 의해 수행되었으며, 완벽성을 보장하지 않습니다.\n* AI는 데이터와 학습 방식에 따라 편향된 결과를 제시할 수도 있습니다.\n* 제공된 분석은 참고 자료로 활용하시고, 최종적인 판단은 사용자의 책임하에 이루어져야 합니다. """)
    framing_result = detect_bias_openai(title_to_display, text_content)
    st.info(framing_result)

# --- newspaper3k Config 객체 (전역 또는 필요시 생성) ---
NEWS_CONFIG = Config()
NEWS_CONFIG.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
NEWS_CONFIG.request_timeout = 15

# --- Streamlit 앱 UI 구성 ---
st.set_page_config(page_title="뉴스읽은척방지기 (하이브리드 AI)", page_icon="🧐")
st.title("🧐 뉴스읽은척방지기")
st.write("키워드 검색 또는 URL 직접 입력으로 뉴스 기사를 AI와 함께 분석해보세요!")
st.caption("본문 요약 및 키워드 추출은 Gemini AI, 프레이밍 분석은 OpenAI GPT를 사용합니다.")

input_tab1, input_tab2 = st.tabs(["🗂️ 키워드/RSS피드로 뉴스 검색/분석", "🔗 URL 직접 입력/분석"])

with input_tab1:
    st.subheader("키워드 또는 RSS 피드 URL로 뉴스 찾아 분석하기")
    search_type_tab1 = st.radio( "검색/입력 타입 선택:", ("키워드로 Google News 검색", "RSS 피드 URL 직접 입력"), key="search_type_tab1", horizontal=True)

    if search_type_tab1 == "키워드로 Google News 검색":
        input_label_tab1 = "검색할 키워드를 입력하세요:"
        input_placeholder_tab1 = "예: 애플 AI 전략"
    else: 
        input_label_tab1 = "뉴스 RSS 피드의 전체 URL을 입력하세요:"
        input_placeholder_tab1 = "예: https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml"
    rss_or_keyword_input_tab1 = st.text_input(input_label_tab1, placeholder=input_placeholder_tab1, key="rss_or_keyword_input_tab1")

    if st.button("📰 뉴스 목록 가져오기", key="fetch_list_button_tab1", use_container_width=True):
        article_options_tab1 = {} 
        if not rss_or_keyword_input_tab1:
            st.warning("키워드 또는 RSS 피드 URL을 입력해주세요.")
        else:
            feed_url_to_parse = None 
            if search_type_tab1 == "RSS 피드 URL 직접 입력":
                if not (rss_or_keyword_input_tab1.startswith('http://') or rss_or_keyword_input_tab1.startswith('https://')):
                    st.warning("올바른 RSS 피드 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작해야 합니다.")
                else:
                    feed_url_to_parse = rss_or_keyword_input_tab1
                feed_source_name = "입력하신 RSS 피드"
            elif search_type_tab1 == "키워드로 Google News 검색":
                feed_url_to_parse = f"https://news.google.com/rss/search?q={rss_or_keyword_input_tab1}&hl=ko&gl=KR&ceid=KR:ko"
                feed_source_name = f"'{rss_or_keyword_input_tab1}' 관련 Google News"

            if feed_url_to_parse: 
                try:
                    with st.spinner(f"{feed_source_name}에서 뉴스를 가져오는 중..."):
                        feed = feedparser.parse(feed_url_to_parse)
                    if feed.entries:
                        for entry in feed.entries[:30]: 
                            if hasattr(entry, 'title') and hasattr(entry, 'link'):
                                article_options_tab1[entry.title] = entry.link
                        if article_options_tab1:
                             st.success(f"{feed_source_name}에서 {len(article_options_tab1)}건의 기사 제목을 찾았습니다.")
                        else:
                            st.warning(f"{feed_source_name}에서 기사를 찾을 수 없거나, 기사 형식이 올바르지 않습니다.")
                    else:
                        st.warning(f"{feed_source_name}에서 기사를 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"{search_type_tab1} 처리 중 오류 발생: {e}")
        
        if article_options_tab1:
            st.session_state.article_options_for_analysis_tab1 = article_options_tab1
        else: 
            if 'article_options_for_analysis_tab1' in st.session_state:
                del st.session_state.article_options_for_analysis_tab1

    if 'article_options_for_analysis_tab1' in st.session_state and st.session_state.article_options_for_analysis_tab1:
        selected_title_tab1 = st.selectbox(
            "분석할 기사를 선택하세요:",
            options=list(st.session_state.article_options_for_analysis_tab1.keys()),
            index=None,
            placeholder="목록에서 기사를 선택하세요...",
            key="selectbox_tab1"
        )
        if selected_title_tab1 and st.button("👆 선택한 뉴스 분석하기", key="analyze_selected_button_tab1", use_container_width=True):
            url_to_analyze_initially = st.session_state.article_options_for_analysis_tab1[selected_title_tab1]
            st.info(f"선택한 기사 분석 중: {selected_title_tab1}")
            
            actual_url_to_process = url_to_analyze_initially 
            if "news.naver.com" in url_to_analyze_initially:
                with st.spinner("Naver 뉴스 기사 원문 링크를 찾는 중..."):
                    original_url = get_original_url_from_naver_news(url_to_analyze_initially)
                if original_url:
                    st.info(f"Naver 뉴스에서 추출된 원문 링크: {original_url}")
                    actual_url_to_process = original_url
                else:
                    st.warning("Naver 뉴스에서 원문 링크를 추출하지 못했습니다. Naver 링크로 분석을 시도합니다.")
            
            try:
                with st.spinner(f"'{selected_title_tab1}' 기사를 가져와 AI가 분석 중입니다... (URL: {actual_url_to_process})"):
                    article = Article(actual_url_to_process, config=NEWS_CONFIG, language='ko') 
                    article.download()
                    article.parse()
                    if not article.title or not article.text or len(article.text) < 50:
                        st.error("선택한 기사의 제목이나 본문을 가져오지 못했거나 내용이 너무 짧습니다.")
                    else:
                        title_for_analysis = article.title if article.title else selected_title_tab1 
                        display_article_analysis_content(title_for_analysis, article.text, actual_url_to_process)
            except Exception as e:
                st.error(f"선택한 기사 처리 중 오류 발생: {e}")

with input_tab2:
    st.subheader("URL로 직접 뉴스 분석하기")
    url_direct_input_tab2 = st.text_input("분석할 뉴스 기사의 전체 URL을 입력해주세요:", placeholder="예: https://www.example-news.com/news/article123", key="url_direct_input_tab2")

    if st.button("🚀 URL 분석 시작", use_container_width=True, key="direct_url_analyze_button_tab2"):
        if not url_direct_input_tab2:
            st.warning("분석할 기사의 URL을 입력해주세요.")
        elif not (url_direct_input_tab2.startswith('http://') or url_direct_input_tab2.startswith('https://')):
            st.warning("올바른 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작해야 합니다.")
        else:
            st.info(f"입력하신 URL의 기사를 분석합니다: {url_direct_input_tab2}")
            
            actual_url_to_process = url_direct_input_tab2 
            if "news.naver.com" in url_direct_input_tab2:
                with st.spinner("Naver 뉴스 기사 원문 링크를 찾는 중..."):
                    original_url = get_original_url_from_naver_news(url_direct_input_tab2)
                if original_url:
                    st.info(f"Naver 뉴스에서 추출된 원문 링크: {original_url}")
                    actual_url_to_process = original_url
                else:
                    st.warning("Naver 뉴스에서 원문 링크를 추출하지 못했습니다. Naver 링크로 분석을 시도합니다.")
            
            try:
                with st.spinner(f"기사를 가져와 AI가 분석 중입니다... (URL: {actual_url_to_process})"):
                    article = Article(actual_url_to_process, config=NEWS_CONFIG, language='ko')
                    article.download()
                    article.parse()
                    if not article.title or not article.text or len(article.text) < 50:
                        st.error("기사 제목이나 본문을 가져오지 못했거나 내용이 너무 짧습니다. 다른 URL을 시도해주세요.")
                    else:
                        display_article_analysis_content(article.title, article.text, actual_url_to_process)
            except Exception as e:
                st.error(f"URL 기사 처리 중 오류 발생: {e}")
                print(f"전체 오류: {e}") 
                st.caption("URL을 확인하시거나, 다른 기사를 시도해보세요. 일부 웹사이트는 외부 접근을 통한 기사 수집을 허용하지 않을 수 있습니다.")