import streamlit as st
import streamlit.components.v1 as components
import mysql.connector
from mysql.connector import Error
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import platform 
import os 
from PIL import Image
import base64 

logo = Image.open('./assets/logo1.jpg')  # 또는 'assets/logo.png'
def get_base64_image(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image('./assets/logo1.jpg')

# ------------------------------------------------------------------------- # 
# DB 연결 설정


# if platform.system() == 'Linux':
#     from dotenv import load_dotenv
#     load_dotenv()
#     DB_HOST = os.environ.get("DB_HOST")
#     DB_NAME = os.environ.get("DB_NAME") 
#     DB_USER = os.environ.get("DB_USER")
#     DB_PASSWORD = os.environ.get("DB_PASSWORD")

# else: 
#     DB_HOST = st.secrets["database"]["host"]
#     DB_NAME = st.secrets["database"]["database"]
#     DB_USER = st.secrets["database"]["user"]
#     DB_PASSWORD = st.secrets["database"]["password"]

# ------------------------------------------------------------------------- # 
DB_HOST = st.secrets["database"]["host"]
DB_NAME = st.secrets["database"]["database"]
DB_USER = st.secrets["database"]["user"]
DB_PASSWORD = st.secrets["database"]["password"]



# 페이지 설정
st.set_page_config(
    page_title="문학작품 수출패턴 분석 대시보드",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


genre_mapping = {
        "A": "A - 환경·재난",
        "B": "B - 미스터리·스릴러", 
        "C": "C - SF·판타지",
        "D": "D - 사회·정치",
        "E": "E - 이주·전쟁",
        "F": "F - 젠더·다양성",
        "G": "G - 종교·신화",
        "H": "H - 관계·성장",
        "I": "I - 로맨스",
        "J": "J - 역사",
        "미분류": "기타"
    }

# 커스텀 CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg,  #fff3cd 0%, #ffe69c 100%);  # 연한 노랑 그라데이션
    padding: 1rem;
    border-radius: 10px;
    color: #856404;
    text-align: center;
    margin: 0.5rem 0;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 1rem;
    opacity: 0.9;
}

.genre-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# @st.cache_resource 제거!
def get_db_connection():
    """데이터베이스 연결 (매번 새로운 연결)"""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            autocommit=True,  # 추가
            connect_timeout=10,  # 추가: 연결 타임아웃
            sql_mode='TRADITIONAL'  # 추가
        )
        return connection
    except Error as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return None

# @st.cache_resource
# def get_db_connection():
#     """데이터베이스 연결 (리소스 캐시)"""
#     try:
#         connection = mysql.connector.connect(
#             host=DB_HOST,
#             database=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD
#         )
#         return connection
#     except Error as e:
#         st.error(f"데이터베이스 연결 오류: {e}")
#         return None

@st.cache_data(ttl=300)
def load_literature_data():
    """literature_books 테이블에서 대시보드용 데이터 로드"""
    connection = None  # 초기화
    cursor = None
    
    try:
        connection = get_db_connection()  # 매번 새로운 연결
        if not connection:
            return None
        
        # 연결 상태 확인 추가
        if not connection.is_connected():
            st.error("데이터베이스 연결이 끊어졌습니다.")
            return None
            
        queries = {
            # 1. 총 원작 작품 수 (원작여부가 'original'인 것들)
            'total_originals': """
                SELECT COUNT(DISTINCT 원작_제목) as count 
                FROM literature_books
                WHERE 원작여부 = 'original'
            """,
            
            # 2. 총 에디션 수
            'total_editions': """
                SELECT COUNT(*) as count 
                FROM literature_books
            """,
            
            # 3. 수출 국가 수
            'total_countries': """
                SELECT COUNT(DISTINCT 국가) as count 
                FROM literature_books 
                WHERE 국가 IS NOT NULL AND 국가 != ''
            """,
            
            # 4. 장르별 분포 (상위 10개)
            'genre_distribution': """
                SELECT 
                    genre1,
                    COUNT(*) as count
                FROM literature_books 
                WHERE genre1 IS NOT NULL AND genre1 != ''
                GROUP BY genre1
                ORDER BY count DESC 
                LIMIT 10
            """,
            
            # 5. 국가별 분포 (상위 15개)
            'country_distribution': """
                SELECT 
                    국가,
                    COUNT(*) as count
                FROM literature_books 
                WHERE 국가 IS NOT NULL AND 국가 != ''
                AND 원작여부 = 'edition'   
                GROUP BY 국가
                ORDER BY count DESC 
                LIMIT 15
            """,
            
            # 5-2. 원작 국가별 분포 (상위 15개)
            'original_country_distribution': """
                SELECT 
                    국가,
                    COUNT(DISTINCT 원작_제목) as count
                FROM literature_books 
                WHERE 국가 IS NOT NULL AND 국가 != ''
                AND 원작여부 = 'original'   
                GROUP BY 국가
                ORDER BY count DESC 
                LIMIT 15
            """,
            
            # 6. 연도별 트렌드 (원작과 에디션)
            'yearly_trend': """
                SELECT 
                    year,
                    COUNT(DISTINCT CASE WHEN 원작여부 = 'original' THEN 원작_제목 END) as originals,
                    COUNT(*) as editions
                FROM literature_books 
                WHERE year IS NOT NULL
                AND year BETWEEN 2016 AND YEAR(CURDATE())
                GROUP BY year
                ORDER BY year
            """,
            
            # 7. 언어별 분포 (상위 10개)
            'language_distribution': """
                SELECT 
                    언어,
                    COUNT(*) as count
                FROM literature_books 
                WHERE 언어 IS NOT NULL AND 언어 != ''
                GROUP BY 언어
                ORDER BY count DESC 
                LIMIT 10
            """,
            # 기존 queries 딕셔너리에 추가
            'country_yearly_trend': """
                WITH original_countries AS (
                    SELECT DISTINCT 원작_제목, 국가 as 원작국가
                    FROM literature_books 
                    WHERE 원작여부 = 'original'
                )
                SELECT 
                    oc.원작국가,
                    lb.year,
                    COUNT(DISTINCT CASE WHEN lb.원작여부 = 'original' THEN lb.원작_제목 END) as originals,
                    COUNT(CASE WHEN lb.원작여부 = 'edition' THEN 1 END) as editions
                FROM literature_books lb
                JOIN original_countries oc ON lb.원작_제목 = oc.원작_제목
                WHERE lb.year IS NOT NULL
                AND lb.year BETWEEN 2016 AND YEAR(CURDATE())
                AND oc.원작국가 IS NOT NULL AND oc.원작국가 != ''
                GROUP BY oc.원작국가, lb.year
                ORDER BY oc.원작국가, lb.year
            """
        }
        
        
        results = {}
        cursor = connection.cursor()
        
        for key, query in queries.items():
            try:
                cursor.execute(query)
                if key in ['total_originals', 'total_editions', 'total_countries']:
                    result = cursor.fetchone()
                    results[key] = result[0] if result and result[0] is not None else 0
                else:
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=columns)
                    
                    if key == 'genre_distribution':
                        df['genre1'] = df['genre1'].map(genre_mapping).fillna(df['genre1'])
                    
                    results[key] = df
                            
            except Error as e:
                st.warning(f"쿼리 실행 중 오류 ({key}): {e}")
                if key in ['total_originals', 'total_editions', 'total_countries']:
                    results[key] = 0
                else:
                    results[key] = pd.DataFrame()
        
        return results
        
    except Error as e:
        st.error(f"데이터 로드 오류: {e}")
        return None
    finally:
        # 안전한 연결 종료
        try:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()
        except:
            pass  # 종료 시 에러 무시

# @st.cache_data(ttl=300)
# def load_literature_data():
#     """literature_books 테이블에서 대시보드용 데이터 로드"""
#     try:
#         connection = get_db_connection()
#         if not connection:
#             return None
        
#         queries = {
#             # 1. 총 원작 작품 수 (원작여부가 'original'인 것들)
#             'total_originals': """
#                 SELECT COUNT(DISTINCT 원작_제목) as count 
#                 FROM literature_books
#                 WHERE 원작여부 = 'original'
#             """,
            
#             # 2. 총 에디션 수
#             'total_editions': """
#                 SELECT COUNT(*) as count 
#                 FROM literature_books
#             """,
            
#             # 3. 수출 국가 수
#             'total_countries': """
#                 SELECT COUNT(DISTINCT 국가) as count 
#                 FROM literature_books 
#                 WHERE 국가 IS NOT NULL AND 국가 != ''
#             """,
            
#             # 4. 장르별 분포 (상위 10개)
#             'genre_distribution': """
#                 SELECT 
#                     genre1,
#                     COUNT(*) as count
#                 FROM literature_books 
#                 WHERE genre1 IS NOT NULL AND genre1 != ''
#                 GROUP BY genre1
#                 ORDER BY count DESC 
#                 LIMIT 10
#             """,
            
#             # 5. 국가별 분포 (상위 15개)
#             'country_distribution': """
#                 SELECT 
#                     국가,
#                     COUNT(*) as count
#                 FROM literature_books 
#                 WHERE 국가 IS NOT NULL AND 국가 != ''
#                 AND 원작여부 = 'edition'   
#                 GROUP BY 국가
#                 ORDER BY count DESC 
#                 LIMIT 15
#             """,
            
#             # 6. 연도별 트렌드 (원작과 에디션)
#             'yearly_trend': """
#                 SELECT 
#                     year,
#                     COUNT(DISTINCT CASE WHEN 원작여부 = 'original' THEN 원작_제목 END) as originals,
#                     COUNT(*) as editions
#                 FROM literature_books 
#                 WHERE year IS NOT NULL
#                 AND year BETWEEN 2016 AND YEAR(CURDATE())
#                 GROUP BY year
#                 ORDER BY year
#             """,
            
#             # 7. 언어별 분포 (상위 10개)
#             'language_distribution': """
#                 SELECT 
#                     언어,
#                     COUNT(*) as count
#                 FROM literature_books 
#                 WHERE 언어 IS NOT NULL AND 언어 != ''
#                 GROUP BY 언어
#                 ORDER BY count DESC 
#                 LIMIT 10
#             """,
#             # 기존 queries 딕셔너리에 추가
#             'country_yearly_trend': """
#                 WITH original_countries AS (
#                     SELECT DISTINCT 원작_제목, 국가 as 원작국가
#                     FROM literature_books 
#                     WHERE 원작여부 = 'original'
#                 )
#                 SELECT 
#                     oc.원작국가,
#                     lb.year,
#                     COUNT(DISTINCT CASE WHEN lb.원작여부 = 'original' THEN lb.원작_제목 END) as originals,
#                     COUNT(CASE WHEN lb.원작여부 = 'edition' THEN 1 END) as editions
#                 FROM literature_books lb
#                 JOIN original_countries oc ON lb.원작_제목 = oc.원작_제목
#                 WHERE lb.year IS NOT NULL
#                 AND lb.year BETWEEN 2016 AND YEAR(CURDATE())
#                 AND oc.원작국가 IS NOT NULL AND oc.원작국가 != ''
#                 GROUP BY oc.원작국가, lb.year
#                 ORDER BY oc.원작국가, lb.year
#             """
#         }
        
#         results = {}
#         cursor = connection.cursor()
        
#         for key, query in queries.items():
#             try:
#                 cursor.execute(query)
#                 if key in ['total_originals', 'total_editions', 'total_countries']:
#                     result = cursor.fetchone()
#                     results[key] = result[0] if result and result[0] is not None else 0
#                 else:
#                     columns = [desc[0] for desc in cursor.description]
#                     data = cursor.fetchall()
#                     df = pd.DataFrame(data, columns=columns)
                    
#                     # 장르 데이터인 경우 매핑 적용
#                     if key == 'genre_distribution':
#                         df['genre1'] = df['genre1'].map(genre_mapping).fillna(df['genre1'])
                    
#                     results[key] = df
                            
#             except Error as e:
#                 st.warning(f"쿼리 실행 중 오류 ({key}): {e}")
#                 # 기본값 설정
#                 if key in ['total_originals', 'total_editions', 'total_countries']:
#                     results[key] = 0
#                 else:
#                     results[key] = pd.DataFrame()
        
#         return results
        
#     except Error as e:
#         st.error(f"데이터 로드 오류: {e}")
#         return None
#     finally:
#         if connection and connection.is_connected():
#             cursor.close()
#             connection.close()

def create_metric_card(title, value, delta=None):
    """메트릭 카드 생성"""
    delta_html = ""
    if delta:
        delta_color = "green" if delta > 0 else "red"
        delta_symbol = "↗" if delta > 0 else "↘"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem;">{delta_symbol} {abs(delta)}</div>'
    
    # 숫자인 경우 콤마 포맷팅, 문자열인 경우 그대로 출력
    if isinstance(value, (int, float)):
        formatted_value = f"{value:,.0f}" if isinstance(value, float) and value.is_integer() else f"{value:,}"
    else:
        formatted_value = str(value)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{formatted_value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def main():
    # st.title("📚 문학작품 수출패턴 분석 대시보드")
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;">
        <h1>문학작품 수출패턴 분석 대시보드</h1>
    </div>
    """, unsafe_allow_html=True)
    st.caption("**Goodreads 베스트셀러 작품(10개년) 기준입니다**")
    st.markdown("---")
    
    # session_state 초기화
    if 'initiated' not in st.session_state:
        st.session_state['initiated'] = False
    
    # 사이드바에 비밀번호 입력
    with st.sidebar.form(key='설정'):
        # --- Secret key input --- #
        secret_key_user = st.text_input(':closed_lock_with_key: **Secret Key**',
                                        placeholder='비밀번호를 입력해주세요.',
                                        type="password")
        # --- Secret key input --- #
        
        submit_prerequisite = st.form_submit_button('**✅ 확인하기**', use_container_width=True)

    if submit_prerequisite:
        if secret_key_user == st.secrets.get("app_password", "your_password"):
            initiated = st.sidebar.success('`Secret Key`가 확인되었습니다', icon="✅")
            st.session_state['initiated'] = True
        else:
            st.sidebar.warning('올바른 `Secret Key`를 입력해 주세요', icon="🚨")
            st.stop()
    
    # 인증 상태에 따른 메시지 표시
    if st.session_state.get('initiated') and not submit_prerequisite:
        st.sidebar.success('`Secret Key`가 확인되었습니다', icon="✅")
    if not st.session_state.get('initiated'):
        st.sidebar.info('`Secret Key`를 입력해 주세요', icon="ℹ️")
        st.stop()
    
    st.sidebar.markdown("---")
    
    
    
    # 데이터 로드
    with st.spinner("literature_books 데이터를 불러오는 중..."):
        data = load_literature_data()
    
    if not data:
        st.error("literature_books 테이블에서 데이터를 불러올 수 없습니다.")
        st.info("데이터베이스 연결이나 테이블 구조를 확인해주세요.")
        return
    
    # 1. 핵심 KPI 섹션
    st.subheader("📊 기본 데이터 설명")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("총 원작 작품 수", data['total_originals'])
    
    with col2:
        create_metric_card("총 에디션 수", data['total_editions'])
    
    with col3:
        create_metric_card("수출 국가 수", data['total_countries'])
    
    with col4:
        ratio = data['total_editions'] / data['total_originals'] if data['total_originals'] > 0 else 0
        create_metric_card("에디션/원작 비율", f"{ratio:.1f}")
    
    st.markdown("---")
    
    # 2. 시각화 섹션 - 첫 번째 행: 장르별 분포 + 번역 언어 분포
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 베스트셀러 장르 분포")
        if not data['genre_distribution'].empty:
            fig_genre = px.pie(
                data['genre_distribution'], 
                values='count', 
                names='genre1',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_genre.update_traces(textposition='inside', textinfo='percent+label')
            fig_genre.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_genre, use_container_width=True)
        else:
            st.info("장르 데이터가 없습니다.")
    
    with col2:
        st.subheader("🗣️ 번역 언어 분포")
        if not data['language_distribution'].empty:
            fig_lang = px.bar(
                data['language_distribution'],
                x='언어',
                y='count',
                color='count',
                color_continuous_scale='Viridis'
            )
            fig_lang.update_layout(height=400)
            st.plotly_chart(fig_lang, use_container_width=True)
        else:
            st.info("언어별 데이터가 없습니다.")
    
    # 3. 두 번째 행: 원작 국가 분포 + 주요 수출 대상국
    col3, col4 = st.columns(2)
    
    # with col3:
    #     st.subheader("🏠 원작 국가 분포")
    #     if not data['original_country_distribution'].empty:
    #         fig_original_country = px.bar(
    #             data['original_country_distribution'], 
    #             x='count', 
    #             y='국가',
    #             orientation='h',
    #             color='count',
    #             color_continuous_scale='Greens'
    #         )
    #         fig_original_country.update_layout(
    #             height=400,
    #             yaxis={'categoryorder': 'total ascending'}
    #         )
    #         st.plotly_chart(fig_original_country, use_container_width=True)
    #     else:
    #         st.info("원작 국가별 데이터가 없습니다.")

    with col3:
        st.subheader("🏠 원작 국가 분포")
        if not data['original_country_distribution'].empty:
            df_country = data['original_country_distribution'].copy()
            
            # 미국만 표시값을 500으로 제한
            df_country['display_count'] = df_country.apply(
                lambda row: 500 if row['국가'] == '미국' else row['count'],
                axis=1
            )
            
            fig_original_country = px.bar(
                df_country, 
                x='display_count', 
                y='국가',
                orientation='h',
                color='count',  # 원본 값으로 색상은 유지
                color_continuous_scale='Greens',
                hover_data={
                    'count': ':,',  # 실제 값을 쉼표와 함께 표시
                    'display_count': False  # 표시값은 호버에서 숨김
                }
            )
            
            # 호버 템플릿 커스터마이징 (선택사항)
            fig_original_country.update_traces(
                hovertemplate='<b>%{y}</b><br>개수: %{customdata[0]:,}<extra></extra>'
            )
            
            fig_original_country.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='count'  # x축 제목을 'count'로 설정
            )
            
            st.plotly_chart(fig_original_country, use_container_width=True)
            
            # # 미국 실제 개수 표시
            # us_actual = df_country[df_country['국가'] == '미국']['count'].iloc[0] if '미국' in df_country['국가'].values else 0
            # if us_actual > 500:
                # st.caption(f"📍 미국 실제 개수: {us_actual:,}개 (차트에서는 500으로 표시)")
                
        else:
            st.info("원작 국가별 데이터가 없습니다.")  

    with col4:
        st.subheader("🌍 주요 수출 대상국")
        if not data['country_distribution'].empty:
            fig_country = px.bar(
                data['country_distribution'], 
                x='count', 
                y='국가',
                orientation='h',
                color='count',
                color_continuous_scale='Blues'
            )
            fig_country.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.info("수출 대상국별 데이터가 없습니다.")
    
    # 4. 세 번째 행: 연도별 트렌드
    st.subheader("📈 연도별 수출 트렌드")
    
    if not data['yearly_trend'].empty:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(
                x=data['yearly_trend']['year'],
                y=data['yearly_trend']['originals'],
                mode='lines+markers',
                name='원작',
                line=dict(color='#667eea', width=3)
            ),
            secondary_y=False,
        )
        
        fig_trend.add_trace(
            go.Scatter(
                x=data['yearly_trend']['year'],
                y=data['yearly_trend']['editions'],
                mode='lines+markers',
                name='에디션',
                line=dict(color='#764ba2', width=3)
            ),
            secondary_y=True,
        )
        
        fig_trend.update_xaxes(title_text="연도")
        fig_trend.update_yaxes(title_text="원작 수", secondary_y=False)
        fig_trend.update_yaxes(title_text="에디션 수", secondary_y=True)
        fig_trend.update_layout(
            height=400, 
            hovermode='x unified',
            title="전체 수출 트렌드"
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("연도별 트렌드 데이터가 없습니다.")
    
    # 5. 사이드바 정보
    with st.sidebar:
        st.header("📊 대시보드 정보")
        # st.markdown("**데이터 소스**: `literature_books`")
        st.markdown(f"- 원작 작품: {data['total_originals']:,}개")
        st.markdown(f"- 에디션: {data['total_editions']:,}개") 
        st.markdown(f"- 수출 국가: {data['total_countries']}개국")
        
        st.markdown("---")
        
        # 새로고침 버튼
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()