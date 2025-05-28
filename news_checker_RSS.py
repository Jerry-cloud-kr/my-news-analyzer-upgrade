import streamlit as st
from newspaper import Article, Config
from sentence_transformers import SentenceTransformer, util
import openai
from openai import OpenAI
import google.generativeai as genai
import feedparser
import requests

# --- API Key 및 클라이언트 설정 (Secrets 우선) ---
client_openai = None 
OPENAI_API_KEY_Direct_Placeholder = "YOUR_OPENAI_KEY_PLACEHOLDER" 
try:
    OPENAI_API_KEY_FROM_SECRETS = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY_FROM_SECRETS:
         st.error("⚠️ OpenAI API 키가 Streamlit Secrets에 설정되었으나 값이 비어있습니다.")
         st.stop()
    client_openai = OpenAI(api_key=OPENAI_API_KEY_FROM_SECRETS) 
except KeyError:
    if OPENAI_API_KEY_Direct_Placeholder == "YOUR_OPENAI_KEY_PLACEHOLDER" or not OPENAI_API_KEY_Direct_Placeholder:
        st.error("OpenAI API 키를 Secrets에서 찾을 수 없습니다. 로컬 테스트 시 코드 상단 플레이스홀더에 실제 키를 입력해주세요.")
        st.stop()
    else: 
        st.warning("로컬 테스트용 OpenAI API 키가 코드에 직접 설정되어 있습니다. GitHub 푸시 전 반드시 Secrets 방식으로 변경하세요.", icon="❗")
        client_openai = OpenAI(api_key=OPENAI_API_KEY_Direct_Placeholder)
except Exception as e:
    st.error(f"OpenAI API 키/클라이언트 설정 오류: {e}")
    st.stop()
if client_openai is None: 
    st.error("OpenAI 클라이언트 초기화 실패. API 키를 확인하세요.")
    st.stop()

GOOGLE_AI_API_KEY_Direct_Placeholder = "YOUR_GOOGLE_AI_KEY_PLACEHOLDER" 
try:
    GOOGLE_AI_API_KEY_FROM_SECRETS = st.secrets["GOOGLE_AI_API_KEY"]
    if not GOOGLE_AI_API_KEY_FROM_SECRETS:
         st.error("⚠️ Google AI API 키가 Streamlit Secrets에 설정되었으나 값이 비어있습니다.")
         st.stop()
    genai.configure(api_key=GOOGLE_AI_API_KEY_FROM_SECRETS)
except KeyError:
    if GOOGLE_AI_API_KEY_Direct_Placeholder == "YOUR_GOOGLE_AI_KEY_PLACEHOLDER" or not GOOGLE_AI_API_KEY_Direct_Placeholder:
        st.error("Google AI API 키를 Secrets에서 찾을 수 없습니다. 로컬 테스트 시 코드 상단 플레이스홀더에 실제 키를 입력해주세요.")
        st.stop()
    else: 
        st.warning("로컬 테스트용 Google AI API 키가 코드에 직접 설정되어 있습니다. GitHub 푸시 전 반드시 Secrets 방식으로 변경하세요.", icon="❗")
        genai.configure(api_key=GOOGLE_AI_API_KEY_Direct_Placeholder)
except Exception as e:
    st.error(f"Google AI API 키 설정 오류: {e}")
    st.stop()

# --- Helper Functions ---
@st.cache_data 
def get_final_url(url, timeout=10):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.url
    except requests.exceptions.RequestException as e:
        print(f"최종 URL 요청 중 오류 ({url}): {e}") # 콘솔 로그는 유지
        return url 
    except Exception as e:
        print(f"최종 URL 확인 중 기타 오류 ({url}): {e}") # 콘솔 로그는 유지
        return url

# --- AI 기능 함수들 --- (내용은 이전과 동일)
def summarize_text_gemini(text_content):
    model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', system_instruction="너는 뉴스 기사의 핵심 내용을 객관적으로 요약하는 AI야.")
    prompt = f"다음 뉴스 기사 본문을 객관적인 사실에 기반하여 핵심 내용 중심으로 요약해 주십시오. 요약에는 주요 인물, 발생한 사건, 중요한 발언, 그리고 사건의 배경 정보가 포함되어야 합니다. 주관적인 해석, 평가, 또는 기사에 명시적으로 드러나지 않은 추론은 배제하고, 사실 관계를 명확히 전달하는 데 집중해 주십시오. 분량은 한국어 기준으로 약 3~5문장 (또는 100~150 단어) 정도로 간결하게 작성해 주십시오.\n\n기사:\n{text_content}"
    try:
        response = model.generate_content(prompt,generation_config=genai.types.GenerationConfig(temperature=0.3))
        return response.text.strip()
    except Exception as e:
        st.warning("Gemini 요약 생성 중 오류가 발생했습니다.")
        print(f"Gemini 요약 API 오류: {e}")
        return "요약 생성에 실패했습니다."

def detect_bias_openai(title, text_content):
    prompt = f"다음은 뉴스 제목과 본문입니다. 제목이 본문 내용을 충분히 반영하고 있는지, 중요한 맥락이나 인물의 입장이 왜곡되거나 누락되었는지 판단해줘.\n\n제목: {title}\n본문: {text_content}\n\n분석 결과를 간단히 3~5줄로 정리해줘."
    try:
        completion = client_openai.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": "너는 공정한 뉴스 프레이밍 분석 도우미야."}, {"role": "user", "content": prompt}])
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning("OpenAI 프레이밍 분석 중 오류가 발생했습니다.")
        print(f"OpenAI 프레이밍 분석 API 오류: {e}")
        return "프레이밍 분석에 실패했습니다."

def extract_keywords_gemini(article_text):
    model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', system_instruction="You are an AI assistant specialized in extracting the most important keywords from news articles. Keywords should be nouns or core noun phrases. Respond only with the keywords, separated by commas.")
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

# --- 유사도 측정 모델 로드 (활성화) ---
model_similarity = None 
try:
    model_similarity = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
    if model_similarity:
        print("SentenceTransformer 모델 로드 성공!") 
    else: 
        st.error("SentenceTransformer 모델 로드에 실패했으나 명시적 오류가 발생하지 않았습니다.")
        # st.stop() # 일단 계속 진행하도록 stop()은 주석처리, 아래에서 model_similarity None 체크
except Exception as e:
    st.error(f"SentenceTransformer 모델 로드 중 심각한 오류 발생: {e}")
    st.warning("유사도 분석 기능이 비활성화될 수 있습니다. 라이브러리 설치 및 호환성을 확인해주세요.")
    # st.stop() # 여기서 멈추지 않고 아래에서 model_similarity가 None인지 확인

# --- 기사 분석 및 결과 표시 함수 ---
def display_article_analysis_content(title_to_display, text_content, article_url):
    st.markdown("---")
    st.subheader("📰 기사 제목")
    st.write(f"**{title_to_display}**")
    st.markdown(f"[🔗 기사 원문 바로가기]({article_url})", unsafe_allow_html=True)
    st.markdown("---")

    # Gemini로 요약
    st.subheader("🧾 본문 요약 (by Gemini AI)")
    with st.expander("⚠️ AI 요약에 대한 중요 안내 (클릭하여 확인)", expanded=False):
        st.markdown(""" **주의: AI 기반 요약 (Gemini)**\n\n* 본 요약은 Gemini 모델을 통해 생성되었으며, 기사의 모든 내용을 완벽하게 반영하지 못할 수 있습니다.\n* AI는 학습 데이터의 한계나 요약 과정의 특성으로 인해 때때로 부정확한 내용을 전달하거나 중요한 내용을 생략할 수 있습니다.\n* 제공된 요약은 기사의 핵심 내용을 빠르게 파악하기 위한 참고 자료로만 활용해주십시오.\n* 기사의 전체적인 맥락과 정확한 정보 확인을 위해서는 반드시 원문 기사를 함께 읽어보시는 것이 중요하며, 최종적인 내용에 대한 판단은 사용자의 책임입니다. """)
    body_summary = summarize_text_gemini(text_content)
    if body_summary == "요약 생성에 실패했습니다.": st.error(body_summary)
    else: st.write(body_summary)
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
    
    # 유사도 판단 (활성화)
    st.subheader("📊 제목-본문요약 유사도 판단")
    if model_similarity is not None: 
        try:
            embeddings = model_similarity.encode([title_to_display, body_summary], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            similarity_threshold_high = 0.65; similarity_threshold_mid = 0.40
            if similarity > similarity_threshold_high: result_text, result_color = "✅ **높음**: 제목이 본문 요약 내용을 잘 반영하고 있습니다.", "green"
            elif similarity > similarity_threshold_mid: result_text, result_color = "🟡 **중간**: 제목이 본문 요약과 다소 관련은 있지만, 내용이 약간 다를 수 있습니다.", "orange"
            else: result_text, result_color = "⚠️ **낮음**: 제목이 본문 요약 내용과 많이 다를 수 있습니다. 낚시성이거나 다른 내용을 다룰 가능성을 확인해보세요.", "red"
            
            st.markdown(f"<span style='color:{result_color};'>{result_text}</span> (유사도 점수: {similarity:.2f})", unsafe_allow_html=True)
            st.caption(f"참고: 유사도는 제목과 AI 요약문 간의 의미적 관계를 나타내며, 임계값(현재: 높음 {similarity_threshold_high}, 중간 {similarity_threshold_mid})에 따라 해석이 달라질 수 있습니다.")
        except Exception as e_sim:
            st.error(f"유사도 분석 중 오류 발생: {e_sim}")
            print(f"유사도 분석 오류: {e_sim}")
            st.info("ℹ️ 유사도 분석을 수행할 수 없습니다.")
    else: 
        st.info("ℹ️ 제목-본문 유사도 분석 기능은 SentenceTransformer 모델 로드 문제로 인해 현재 실행되지 않았습니다.") 
    st.markdown("---")

    # GPT로 프레이밍 분석 (유지)
    st.subheader("🕵️ 프레이밍 분석 결과 (by GPT)")
    with st.expander("⚠️ AI 프레이밍 분석 주의사항 (클릭하여 확인)"):
        st.markdown(""" **주의: AI 기반 프레이밍 분석 (GPT)**\n\n* 본 분석은 GPT 모델에 의해 수행되었으며, 완벽성을 보장하지 않습니다.\n* AI는 데이터와 학습 방식에 따라 편향된 결과를 제시할 수도 있습니다.\n* 제공된 분석은 참고 자료로 활용하시고, 최종적인 판단은 사용자의 책임하에 이루어져야 합니다. """)
    framing_result = detect_bias_openai(title_to_display, text_content)
    if framing_result == "프레이밍 분석에 실패했습니다.": st.error(framing_result)
    else: st.info(framing_result)

# --- newspaper3k Config 객체 ---
NEWS_CONFIG = Config()
NEWS_CONFIG.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
NEWS_CONFIG.request_timeout = 15
NEWS_CONFIG.memoize_articles = False 
NEWS_CONFIG.fetch_images = False 

# --- Streamlit 앱 UI 구성 ---
st.set_page_config(page_title="뉴스읽은척방지기 (AI)", page_icon="🧐")
st.title("🧐 뉴스읽은척방지기")
st.write("키워드 검색 또는 URL 직접 입력으로 뉴스 기사를 AI와 함께 분석해보세요!")
st.caption("본문 요약 및 키워드 추출은 Gemini AI, 프레이밍 분석은 OpenAI GPT, 유사도 분석은 SentenceTransformer를 사용합니다.")

# 입력 방식 선택
if 'current_input_method' not in st.session_state: # session_state 초기화
    st.session_state.current_input_method = "키워드로 Google News 검색"

selected_input_method = st.radio(
    "뉴스 가져오는 방법 선택:",
    ("키워드로 Google News 검색", "URL 직접 입력"),
    key="input_method_selector_radio", # 고유한 key
    horizontal=True,
    index=("키워드로 Google News 검색", "URL 직접 입력").index(st.session_state.current_input_method) # 선택 유지
)
# 라디오 버튼 선택 변경 시 session_state 업데이트
if selected_input_method != st.session_state.current_input_method:
    st.session_state.current_input_method = selected_input_method
    # 선택이 변경되면 하위 위젯들의 상태 유지를 위해 재실행이 필요할 수 있으나,
    # Streamlit은 위젯 변경 시 자동으로 재실행하므로 특별한 처리는 불필요할 수 있음.
    # 만약 selectbox 등의 이전 선택이 꼬이면 st.experimental_rerun() 고려 (하지만 남용 주의)

if st.session_state.current_input_method == "키워드로 Google News 검색":
    st.subheader("🗂️ 키워드로 뉴스 찾아 분석하기")
    search_query = st.text_input("검색할 키워드를 입력하세요:", placeholder="예: 글로벌 경제 동향", key="keyword_search_main_input_field") # key 변경

    if st.button("🔍 뉴스 검색", key="search_button_main_action_key", use_container_width=True): # key 변경
        if not search_query:
            st.warning("검색어를 입력해주세요.")
        else:
            st.session_state.article_options_for_display = None # 이전 결과 초기화
            google_news_rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=ko&gl=KR&ceid=KR:ko"
            fetched_articles_for_display_list = [] 
            try:
                custom_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                with st.spinner(f"'{search_query}' 관련 뉴스를 Google News에서 검색하고 링크를 확인 중..."):
                    feed = feedparser.parse(google_news_rss_url, agent=custom_user_agent)
                    if feed.entries:
                        for entry in feed.entries[:10]: 
                            if hasattr(entry, 'title') and hasattr(entry, 'link'):
                                final_url = get_final_url(entry.link) 
                                if final_url: 
                                    fetched_articles_for_display_list.append({"title": entry.title, "url": final_url, "original_rss_link": entry.link})
                        if fetched_articles_for_display_list:
                             st.success(f"'{search_query}' 관련 뉴스 {len(fetched_articles_for_display_list)}건의 링크를 확인했습니다.")
                        else:
                            st.warning(f"'{search_query}' 관련 Google News에서 유효한 기사 링크를 찾을 수 없거나, 기사 형식이 올바르지 않습니다.")
                    else:
                        st.warning(f"'{search_query}' 관련 Google News에서 기사를 가져오지 못했습니다. (HTTP Status: {feed.get('status', 'N/A')})")
            except Exception as e:
                st.error(f"뉴스 검색 중 오류 발생: {e}")
            
if 'article_options_for_display' in st.session_state and st.session_state.article_options_for_display:
        st.markdown("---")
        
        # st.session_state.article_options_for_display는 [{'title': '제목', 'url': 'URL', 'original_rss_link': '...'}, ...] 형태의 리스트입니다.
        # selectbox에 표시할 제목 목록을 만듭니다. (예: "1. 기사제목 A", "2. 기사제목 B")
        # 그리고 선택된 표시용 제목을 통해 실제 URL을 찾을 수 있도록 딕셔너리도 만듭니다.
        
        display_title_to_url_map = {}
        selectbox_options = ["선택하세요..."] # 드롭다운의 첫 번째 옵션
        
        for i, item in enumerate(st.session_state.article_options_for_display):
            # selectbox에 너무 긴 제목이 들어가지 않도록 처리 (선택적)
            truncated_title = item['title']
            if len(truncated_title) > 80: # 80자 이상이면 줄임
                truncated_title = truncated_title[:80] + "..."
            
            display_option = f"{i+1}. {truncated_title}" # 예: "1. 매우 긴 기사 제목입니다..."
            selectbox_options.append(display_option)
            display_title_to_url_map[display_option] = item['url'] # 표시용 제목과 실제 URL 매핑

        # 이전에 선택한 값을 유지하기 위해 session_state 사용 (선택적)
        if 'selected_article_display_title' not in st.session_state:
            st.session_state.selected_article_display_title = selectbox_options[0] # 기본값 "선택하세요..."

        selected_display_title = st.selectbox(
            "확인할 기사를 선택하세요:",
            options=selectbox_options,
            # index=selectbox_options.index(st.session_state.selected_article_display_title), # 이전 선택 유지 (필요시)
            index = 0, # 항상 "선택하세요..."로 시작
            key="select_article_to_view_url_revised"
        )
        st.session_state.selected_article_display_title = selected_display_title # 현재 선택 저장

        if selected_display_title and selected_display_title != "선택하세요...":
            selected_final_url = display_title_to_url_map[selected_display_title] # 매핑된 실제 URL 가져오기
            
            # 원본 기사 제목 (줄이지 않은 전체 제목)을 다시 찾기 (선택적, 더 정확한 제목 표시 위함)
            original_title_for_selected_url = ""
            for item in st.session_state.article_options_for_display:
                if display_title_to_url_map.get(selected_display_title) == item['url']: # URL이 같은 항목을 찾아
                    original_title_for_selected_url = item['title'] # 원본 제목 사용
                    break
            
            st.markdown(f"**선택한 기사 제목:** {original_title_for_selected_url if original_title_for_selected_url else selected_display_title.split('. ', 1)[-1]}")
            st.markdown(f"**기사 원문 URL (아래 주소를 복사하세요):**")
            st.code(selected_final_url) # URL을 코드 블록으로 보여줘서 복사하기 쉽게
            st.info("👆 위 URL을 복사하여 'URL 직접 입력/분석' 탭에 붙여넣고 분석을 시작하세요.")
        st.markdown("---")

elif st.session_state.current_input_method == "URL 직접 입력":
    st.subheader("🔗 URL 직접 입력하여 분석하기")
    url_direct_input = st.text_input("분석할 뉴스 기사의 전체 URL을 입력해주세요:", placeholder="예: https://www.example-news.com/news/article123", key="url_direct_input_main_field_key") # key 변경

    if st.button("🚀 URL 분석 시작", use_container_width=True, key="direct_url_analyze_button_main_action_key"): # key 변경
        if not url_direct_input:
            st.warning("분석할 기사의 URL을 입력해주세요.")
        elif not (url_direct_input.startswith('http://') or url_direct_input.startswith('https://')):
            st.warning("올바른 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작해야 합니다.")
        else:
            # URL 직접 입력 시에도 get_final_url을 거치도록 하여 일관성 유지
            final_url_to_process = get_final_url(url_direct_input) 
            # st.info(f"입력하신 URL의 기사를 분석합니다: {final_url_to_process}") # 스피너가 대신함
            
            try:
                with st.spinner(f"기사를 가져와 AI가 분석 중입니다... (URL: {final_url_to_process})"):
                    article = Article(final_url_to_process, config=NEWS_CONFIG, language='ko')
                    article.download()
                    article.parse()
                    if not article.title or not article.text or len(article.text) < 50:
                        st.error("기사 제목이나 본문을 가져오지 못했거나 내용이 너무 짧습니다. 다른 URL을 시도해주세요.")
                    else:
                        display_article_analysis_content(article.title, article.text, final_url_to_process)
            except Exception as e:
                st.error(f"URL 기사 처리 중 오류 발생: {e}")
                print(f"전체 오류: {e}") 
                st.caption("URL을 확인하시거나, 다른 기사를 시도해보세요. 일부 웹사이트는 외부 접근을 통한 기사 수집을 허용하지 않을 수 있습니다.")