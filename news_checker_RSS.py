import streamlit as st
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import openai # OpenAI 라이브러리
from openai import OpenAI # OpenAI 클라이언트 클래스 임포트
import google.generativeai as genai
# feedparser는 현재 코드에 import 되어 있으므로 requirements.txt에 포함했습니다.

# --- OpenAI API Key 및 클라이언트 설정 (Secrets 사용) ---
client_openai = None # OpenAI 클라이언트 변수 선언
OPENAI_API_KEY_Direct_Placeholder = "YOUR_OPENAI_KEY_PLACEHOLDER" # 로컬 테스트용 플레이스홀더

try:
    # Streamlit Community Cloud 배포 시 Secrets에 설정된 키를 사용
    OPENAI_API_KEY_FROM_SECRETS = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY_FROM_SECRETS:
         st.error("⚠️ OpenAI API 키가 Streamlit Secrets에 설정되었으나 값이 비어있습니다. 앱 설정에서 확인해주세요.")
         st.stop()
    # openai.api_key = OPENAI_API_KEY_FROM_SECRETS # 전역 설정 (선택적)
    client_openai = OpenAI(api_key=OPENAI_API_KEY_FROM_SECRETS) # Secrets 키로 클라이언트 초기화
except KeyError:
    # 로컬 테스트 시 st.secrets["OPENAI_API_KEY"]가 없을 때
    if OPENAI_API_KEY_Direct_Placeholder == "YOUR_OPENAI_KEY_PLACEHOLDER" or not OPENAI_API_KEY_Direct_Placeholder: # 실제 키가 입력되었는지 확인
        st.error("OpenAI API 키를 Streamlit Secrets에서 찾을 수 없습니다. 로컬 테스트를 원하시면 코드 상단의 OPENAI_API_KEY_Direct_Placeholder 값을 실제 키로 입력하거나, 앱 배포 후 Streamlit Community Cloud의 Secrets 설정을 확인하세요.")
        st.stop()
    else: # 로컬 테스트 시 실제 키가 플레이스홀더에 입력되었다고 가정
        st.warning("로컬 테스트용 OpenAI API 키가 코드에 직접 설정되어 있습니다. GitHub에 배포/푸시하기 전에 이 부분을 반드시 Streamlit Secrets 방식으로 변경하거나 키를 삭제하세요.", icon="❗")
        # openai.api_key = OPENAI_API_KEY_Direct_Placeholder # 전역 설정 (선택적)
        client_openai = OpenAI(api_key=OPENAI_API_KEY_Direct_Placeholder)
except Exception as e:
    st.error(f"OpenAI API 키 설정 또는 클라이언트 초기화 중 오류: {e}")
    st.stop()

if client_openai is None: # client_openai가 어떤 이유로든 초기화되지 않았다면 중단
    st.error("OpenAI 클라이언트가 초기화되지 않았습니다. API 키 설정을 확인해주세요.")
    st.stop()

# --- Google AI API Key 설정 (Secrets 사용) ---
GOOGLE_AI_API_KEY_Direct_Placeholder = "YOUR_GOOGLE_AI_KEY_PLACEHOLDER" # 로컬 테스트용 플레이스홀더
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
    else: # 로컬 테스트 시 실제 키가 플레이스홀더에 입력되었다고 가정
        st.warning("로컬 테스트용 Google AI API 키가 코드에 직접 설정되어 있습니다. GitHub에 배포/푸시하기 전에 이 부분을 반드시 Streamlit Secrets 방식으로 변경하거나 키를 삭제하세요.", icon="❗")
        genai.configure(api_key=GOOGLE_AI_API_KEY_Direct_Placeholder)
except Exception as e:
    st.error(f"Google AI API 키 설정 중 오류: {e}")
    st.stop()


# 요약 함수 (Gemini 사용)
def summarize_text(text):
    # 이 함수 내의 'import google.generativeai as genai'는 스크립트 상단에 이미 있으므로 제거했습니다.
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
    {text}
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
def detect_bias(title, text):
    prompt = f"""
    다음은 뉴스 제목과 본문입니다.
    제목이 본문 내용을 충분히 반영하고 있는지, 중요한 맥락이나 인물의 입장이 왜곡되거나 누락되었는지 판단해줘.

    제목: {title}
    본문: {text}

    분석 결과를 간단히 3~5줄로 정리해줘.
    """
    try:
        completion = client_openai.chat.completions.create(
            model="gpt-4", # 또는 "gpt-4o" 등 사용 가능한 모델
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


# GPT 기반 키워드 추출 함수 (최신 SDK 적용)
def extract_keywords_gpt(article_text):
    prompt = f"""
    다음 뉴스 기사 본문에서 가장 중요한 핵심 키워드를 5개만 추출하여, 각 키워드를 쉼표(,)로 구분한 하나의 문자열로 응답해줘. 다른 설명이나 문장은 포함하지 마.

    기사 본문:
    {article_text}
    """
    try:
        completion = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 뉴스 키워드 추출을 잘하는 요약봇이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        keywords_string = completion.choices[0].message.content.strip()
        if ":" in keywords_string: # 간단한 후처리
            keywords_string = keywords_string.split(":")[-1].strip()
        return [kw.strip() for kw in keywords_string.split(',') if kw.strip()]
    except Exception as e:
        st.warning("AI 키워드 추출 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        print(f"OpenAI 키워드 추출 API 오류: {e}")
        return []

# 유사도 측정 모델 로드
try:
    model_similarity = SentenceTransformer('all-MiniLM-L6-v2') # 변수명 변경 (model -> model_similarity)
except Exception as e:
    st.error(f"SentenceTransformer 모델 로드 중 오류: {e}")
    st.stop()


# Streamlit 인터페이스 시작
st.set_page_config(page_title="뉴스읽은척방지기 (하이브리드)", page_icon="🧐")
st.title("🧐 뉴스읽은척방지기")
st.write("기사 제목이 본문과 어울리는지, 왜곡됐는지 AI와 함께 분석해보자!")
st.caption("본문 요약은 Gemini AI, 키워드 추출 및 프레이밍 분석은 OpenAI GPT를 사용합니다.")


url = st.text_input("뉴스 기사 URL을 입력하세요:", placeholder="예: https://www.example.com/news/article-link")

# 버튼은 한 번만 생성합니다.
if st.button("📰 기사 분석 시작", use_container_width=True, key="analyze_button_main"):
    if not url:
        st.warning("뉴스 기사 URL을 입력해주세요.")
    elif not (url.startswith('http://') or url.startswith('https://')):
        st.warning("올바른 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작해야 합니다.")
    else:
        try:
            with st.spinner("기사를 가져와 AI가 분석 중입니다... 잠시 기다려주세요."):
                # ... (이하 기존 분석 로직 그대로) ...
                article = Article(url, language='ko')
                article.download()
                article.parse()

                title = article.title
                text = article.text

                if not title or not text or len(text) < 50:
                    st.error("기사 제목이나 본문을 가져오지 못했거나 내용이 너무 짧습니다. 다른 URL을 시도해주세요.")
                else:
                    st.markdown("---")
                    st.subheader("📰 기사 제목")
                    st.write(f"**{title}**")
                    st.markdown(f"[🔗 기사 원문 바로가기]({url})", unsafe_allow_html=True)
                    st.markdown("---")

                    # Gemini로 요약
                    st.subheader("🧾 본문 요약 (by Gemini AI)")
                    with st.expander("⚠️ AI 요약에 대한 중요 안내 (클릭하여 확인)"):
                        st.markdown("""
                        - 본 요약은 **Gemini 모델**을 통해 생성되었습니다.
                        - 모든 내용을 완벽히 반영하지 못할 수 있으며, 중요한 내용은 원문을 통해 확인하시는 것이 좋습니다. 최종 판단은 사용자에게 달려 있습니다.
                        """)
                    body_summary = summarize_text(text) # Gemini 요약 함수 호출
                    st.write(body_summary)
                    st.markdown("---")

                    # GPT로 키워드 추출 및 비교
                    st.subheader("🔍 AI 추출 주요 키워드와 제목 비교 (by GPT)")
                    extracted_keywords = extract_keywords_gpt(text) # GPT 키워드 추출 함수 호출
                    if not extracted_keywords:
                        st.info("ℹ️ AI가 본문에서 주요 키워드를 추출하지 못했거나, 추출된 키워드가 없습니다.")
                    else:
                        st.caption(f"AI(GPT)가 본문에서 추출한 주요 키워드: **{', '.join(extracted_keywords)}**")
                        missing_in_title = [kw for kw in extracted_keywords if kw.lower() not in title.lower()]
                        if missing_in_title:
                            st.warning(f"❗ AI 추출 키워드 중 일부가 제목에 빠져있을 수 있습니다: **{', '.join(missing_in_title)}**")
                        else:
                            st.success("✅ AI 추출 핵심 키워드가 제목에 잘 반영되어 있습니다.")
                    st.markdown("---")
                    
                    # 유사도 판단
                    st.subheader("📊 제목-본문요약 유사도 판단")
                    embeddings = model_similarity.encode([title, body_summary], convert_to_tensor=True) # model_similarity 사용
                    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                    
                    similarity_threshold_high = 0.65
                    similarity_threshold_mid = 0.40
                    if similarity > similarity_threshold_high:
                        result_text = "✅ **높음**: 제목이 AI 요약 내용을 잘 반영하고 있습니다."
                        result_color = "green"
                    elif similarity > similarity_threshold_mid:
                        result_text = "🟡 **중간**: 제목이 AI 요약과 다소 관련은 있지만, 내용이 약간 다를 수 있습니다."
                        result_color = "orange"
                    else:
                        result_text = "⚠️ **낮음**: 제목이 AI 요약 내용과 많이 다를 수 있습니다."
                        result_color = "red"
                    st.markdown(f"<span style='color:{result_color};'>{result_text}</span> (유사도 점수: {similarity:.2f})", unsafe_allow_html=True)
                    st.markdown("---")

                    # GPT로 프레이밍 분석
                    st.subheader("🕵️ 프레이밍 분석 결과 (by GPT)")
                    with st.expander("⚠️ AI 프레이밍 분석 주의사항 (클릭하여 확인)"):
                        st.markdown("""
                        - 본 분석은 **GPT 모델** 기반이며, 완벽한 해석을 보장하지 않습니다.
                        - 제공된 분석은 참고용이며 최종 판단은 사용자에게 있습니다. AI는 학습 데이터에 따라 편향된 결과를 보일 수 있습니다.
                        """)
                    framing_result = detect_bias(title, text) # GPT 프레이밍 분석 함수 호출
                    st.info(framing_result)

        except Exception as e:
            st.error(f"기사 처리 중 오류 발생: {str(e)}")
            print(f"전체 오류: {e}") 
            st.caption("URL을 확인하시거나, 다른 기사를 시도해보세요. 일부 웹사이트는 외부 접근을 통한 기사 수집을 허용하지 않을 수 있습니다.")

st.markdown("---")
st.subheader("📡 RSS 피드에서 뉴스 가져오기")
rss_url_input = st.text_input("구독하거나 분석하고 싶은 RSS 피드의 URL을 입력하세요:", placeholder="예: https://news.google.com/rss/search?q=인공지능&hl=ko&gl=KR&ceid=KR:ko")

if st.button("📥 RSS 피드에서 기사 불러오기", use_container_width=True):
    if not rss_url_input:
        st.warning("RSS 피드 URL을 입력해주세요.")
    elif not (rss_url_input.startswith('http://') or rss_url_input.startswith('https://')):
        st.warning("올바른 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작해야 합니다.")
    else:
        try:
            with st.spinner(f"'{rss_url_input}' 피드에서 최신 기사를 가져오는 중..."):
                feed = feedparser.parse(rss_url_input)
                if not feed.entries:
                    st.warning("해당 RSS 피드에서 기사를 가져올 수 없거나, 피드에 기사가 없습니다.")
                else:
                    st.info(f"'{feed.feed.title if 'title' in feed.feed else rss_url_input}' 피드에서 {len(feed.entries)}개의 기사를 찾았습니다.")
                    
                    # 기사 제목과 링크를 딕셔너리로 저장 (selectbox에서 제목 선택 시 링크 사용 위함)
                    # 최신 30개 정도만 가져오도록 제한할 수 있습니다.
                    article_options = {entry.title: entry.link for entry in feed.entries[:30]} 
                    
                    if 'selected_article_title_from_rss' not in st.session_state:
                        st.session_state.selected_article_title_from_rss = None

                    # 사용자가 이전에 선택한 항목이 있다면 그것을 기본값으로 설정, 없다면 플레이스홀더
                    # 또는 항상 플레이스홀더를 원하면 index=None 사용 가능 (selectbox 구현에 따라 다름)
                    current_selection = None
                    if st.session_state.selected_article_title_from_rss in article_options:
                         current_selection = st.session_state.selected_article_title_from_rss
                    
                    # selectbox의 key를 고유하게 만들어주면, 다른 selectbox와 충돌 방지
                    selected_title_from_rss = st.selectbox(
                        "분석할 기사를 선택하세요:", 
                        options=list(article_options.keys()), 
                        index=list(article_options.keys()).index(current_selection) if current_selection else 0, # 또는 index=None, placeholder="기사를 선택하세요"
                        key="rss_article_selectbox" 
                    )

                    # 선택된 기사 제목을 session_state에 저장 (선택 유지 위함 - 선택사항)
                    st.session_state.selected_article_title_from_rss = selected_title_from_rss

                    if selected_title_from_rss:
                        st.success(f"선택된 기사: {selected_title_from_rss}")
                        # "선택한 RSS 기사 분석하기" 버튼을 여기에 추가하거나,
                        # 바로 아래 URL 직접 입력칸에 선택된 URL을 채워주고 기존 분석 버튼을 누르게 유도할 수도 있습니다.
                        # 여기서는 분석 버튼을 하나 더 만드는 예시:
                        if st.button("👆 선택한 RSS 기사 분석 실행", key="analyze_rss_button"):
                            # url_input 변수에 선택된 기사의 URL을 할당하여 기존 분석 로직을 태움
                            # 또는 이 버튼 클릭 시 바로 분석 로직을 여기에 구현
                            st.session_state.url_to_analyze = article_options[selected_title_from_rss]
                            st.info(f"분석할 URL이 설정되었습니다: {st.session_state.url_to_analyze}. 상단의 'URL 기사 분석 시작' 버튼을 눌러주세요 (또는 URL 입력창에 자동 입력).")
                            # 더 나은 UX: URL 입력창에 바로 채워주기
                            # (이 부분은 Streamlit에서 입력창 값을 프로그래밍적으로 바꾸는게 간단하지 않으므로,
                            #  분석 로직을 이 버튼 아래에 바로 연결하는게 더 직관적일 수 있습니다.)
                            # 여기서는 st.session_state에 저장하고, 사용자가 메인 분석 버튼을 누르도록 유도하는 대신,
                            # 직접 url_input 값을 변경하는 것은 Streamlit에서 간단하지 않으므로,
                            # 분석 로직을 이 버튼 아래에 직접 연결하는 것이 좋습니다.
                            # 아래는 분석 로직을 바로 연결하는 예시입니다. (기존 분석 버튼 로직을 함수화하면 좋음)

                            # ---- 기존 분석 로직을 여기에 붙여넣거나 함수로 호출 ----
                            # 예시: analyze_article(article_options[selected_title_from_rss])
                            # 지금은 URL 입력창을 사용하는 메인 분석 버튼이 있으므로,
                            # 이 버튼은 URL을 어딘가에 '저장'하고 메인 버튼을 누르도록 안내하는 역할만 하거나,
                            # 메인 분석 로직을 함수로 만들어 여기서도 호출하고 메인 버튼에서도 호출하게 할 수 있습니다.
                            # 일단은, 선택된 URL을 화면에 보여주고, 사용자가 복사해서 위 URL 입력창에 넣고 분석하도록 유도
                            st.code(article_options[selected_title_from_rss])
                            st.info("위 URL을 복사하여 상단 URL 입력창에 붙여넣고 'URL 기사 분석 시작' 버튼을 눌러주세요.")


        except Exception as e:
            st.error(f"RSS 피드를 처리하는 중 오류가 발생했습니다: {str(e)}")
            print(f"RSS 피드 처리 오류: {e}")