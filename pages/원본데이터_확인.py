import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime, date
from io import BytesIO
import platform 
import os 

# ------------------------------------------------------------------------- # 
# DB 연결 설정


def get_secrets():
    """Secrets 정보를 안전하게 가져오는 함수"""
    try:
        # Streamlit Cloud secrets 접근
        return {
            'app_password': st.secrets["app_password"],
            'db_host': st.secrets["database"]["host"],
            'db_name': st.secrets["database"]["database"],
            'db_user': st.secrets["database"]["user"],
            'db_password': st.secrets["database"]["password"]
        }
    except Exception as e:
        st.error(f"Secrets 접근 오류: {e}")
        st.info("Streamlit Cloud Secrets 설정을 확인해주세요.")
        return None


# ------------------------------------------------------------------------- # 

# 함수 호출하여 secrets 가져오기
secrets = get_secrets()

# secrets가 None이 아닐 때만 사용
if secrets:
    correct_password = secrets['app_password']
    DB_HOST = secrets['db_host']
    DB_NAME = secrets['db_name']
    DB_USER = secrets['db_user']
    DB_PASSWORD = secrets['db_password']
else:
    # secrets를 가져올 수 없는 경우 처리
    st.stop()  # 또는 적절한 에러 처리

def get_database_connection():
    """데이터베이스 연결 함수"""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return None

def get_filter_options():
    """필터링에 필요한 옵션들을 데이터베이스에서 조회"""
    connection = get_database_connection()
    if connection is None:
        return None, None, None
    
    try:
        cursor = connection.cursor()
        
        # 국가 목록 조회
        cursor.execute("SELECT DISTINCT 국가 FROM literature_books WHERE 국가 IS NOT NULL ORDER BY 국가")
        countries = [row[0] for row in cursor.fetchall()]
        
        # 장르 목록 조회 (genre1, genre2, genre3, genre4에서)
        cursor.execute("""
            SELECT DISTINCT genre1 FROM literature_books WHERE genre1 IS NOT NULL AND genre1 != ''
            UNION
            SELECT DISTINCT genre2 FROM literature_books WHERE genre2 IS NOT NULL AND genre2 != ''
            UNION
            SELECT DISTINCT genre3 FROM literature_books WHERE genre3 IS NOT NULL AND genre3 != ''
            UNION
            SELECT DISTINCT genre4 FROM literature_books WHERE genre4 IS NOT NULL AND genre4 != ''
            ORDER BY 1
        """)
        genre_codes = [row[0] for row in cursor.fetchall()]
        
        # 연도 범위 조회
        cursor.execute("SELECT MIN(year), MAX(year) FROM literature_books WHERE year IS NOT NULL")
        year_range = cursor.fetchone()
        
        return countries, genre_codes, year_range
        
    except Error as e:
        st.error(f"필터 옵션 조회 오류: {e}")
        return None, None, None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def fetch_literature_data(start_date, end_date, original_filter='all', country_filter='all', 
                         genre_filter='all'): # , year_filter='all'
    """지정된 조건의 문학 도서 데이터 조회"""
    connection = get_database_connection()
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        # 조건들 구성
        conditions = ["발간일 BETWEEN %s AND %s"]
        params = [start_date, end_date]
        
        # 원작여부 필터 조건
        if original_filter == 'original':
            conditions.append("원작여부 = 'original'")
        elif original_filter == 'edition':
            conditions.append("원작여부 = 'edition'")
        
        # 국가 필터 조건
        if country_filter != 'all':
            conditions.append("국가 = %s")
            params.append(country_filter)
        
        # 장르 필터 조건
        if genre_filter != 'all':
            genre_condition = "(genre1 = %s OR genre2 = %s OR genre3 = %s OR genre4 = %s)"
            conditions.append(genre_condition)
            params.extend([genre_filter, genre_filter, genre_filter, genre_filter])
        
        # 연도 필터 조건
        # if year_filter != 'all':
        #     if isinstance(year_filter, tuple):  # 연도 범위인 경우
        #         conditions.append("year BETWEEN %s AND %s")
        #         params.extend(year_filter)
        #     else:  # 특정 연도인 경우
        #         conditions.append("year = %s")
        #         params.append(year_filter)
        
        # 최종 쿼리 구성
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT id, year, 원작_제목, 에디션_제목, 작가명, `ISBN(13)`, ASIN, 
                유형, 출판사명, 언어, 발간일, 수집일자, URL, 국가, 원작여부, 
                genre1, genre2, genre3, genre4
            FROM literature_books 
            WHERE {where_clause}
            ORDER BY 발간일 DESC
            """
            
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # 컬럼명 가져오기
        column_names = [desc[0] for desc in cursor.description]
        
        # DataFrame 생성
        df = pd.DataFrame(results, columns=column_names)
        
        if not df.empty:
            genre_columns = ['genre1', 'genre2', 'genre3', 'genre4']

            # 장르 코드 매핑
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

            # 매핑을 적용하여 장르 변환
            df['장르'] = df[genre_columns].apply(
                lambda row: ', '.join([
                    genre_mapping.get(str(val), str(val)) 
                    for val in row 
                    if pd.notna(val) and val != '' and str(val).strip()
                ]), 
                axis=1
            )

            # 기존 genre 칼럼들 제거
            df = df.drop(columns=genre_columns)

            df = df[['year', '국가', '원작_제목', '작가명', '발간일', 'ISBN(13)', 'ASIN', 
                     '출판사명', '언어', '장르', 'URL', '원작여부']]
        
        return df
        
    except Error as e:
        st.error(f"데이터 조회 오류: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def main():
    st.set_page_config(
        page_title="문학 도서 데이터 조회",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📚 문학 도서 데이터 조회")
    st.markdown("---")
    
    # 🔐 비밀번호 입력
    secret_key_user = st.text_input(':closed_lock_with_key: **Secret Key**',
                                    placeholder='비밀번호를 입력해주세요.',
                                    type="password")
    

    secrets = get_secrets()
    
    if secrets:
        correct_password = secrets['app_password']
        DB_HOST = secrets['db_host']
        DB_NAME = secrets['db_name']
        DB_USER = secrets['db_user']
        DB_PASSWORD = secrets['db_password']
    else:
        st.stop()
    
    # 비밀번호 확인
    if secret_key_user != correct_password:
        st.warning("올바른 비밀번호를 입력해주세요.")
        st.stop()
    

    # 세션 상태 초기화
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'query_info' not in st.session_state:
        st.session_state.query_info = {}
    if 'filter_options_loaded' not in st.session_state:
        st.session_state.filter_options_loaded = False
        st.session_state.countries = []
        st.session_state.genre_codes = []
        st.session_state.year_range = None
    
    # 필터 옵션 로드 (처음 한 번만)
    if not st.session_state.filter_options_loaded:
        with st.spinner("필터 옵션을 불러오는 중..."):
            countries, genre_codes, year_range = get_filter_options()
            if countries is not None:
                st.session_state.countries = countries
                st.session_state.genre_codes = genre_codes
                st.session_state.year_range = year_range
                st.session_state.filter_options_loaded = True
    
    # 사이드바에 모든 필터링 조건 설정
    st.sidebar.header("🔍 조회 및 필터링 조건")
    
    # 기간 설정
    st.sidebar.subheader("📅 기간 설정")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일", 
            value=date(2015, 1, 1),
            help="조회 시작 날짜를 선택하세요"
        )
    
    with col2:
        end_date = st.date_input(
            "종료일", 
            value=date.today(),
            help="조회 종료 날짜를 선택하세요"
        )
    
    # 날짜 유효성 검사
    if start_date > end_date:
        st.sidebar.error("시작일이 종료일보다 늦을 수 없습니다.")
        return
    
    st.sidebar.markdown("---")
    
    # 원작여부 선택
    st.sidebar.subheader("📖 원작여부")
    original_filter = st.sidebar.selectbox(
        "원작여부 선택",
        index=1,  # 기본값 설정 (0='all', 1='original', 2='edition')
        options=['all', 'original', 'edition'],
        format_func=lambda x: {'all': '전체', 'original': '원작', 'edition': '에디션'}[x],
        help="조회할 데이터 유형을 선택하세요"
    )
    
    # 국가 선택
    st.sidebar.subheader("🌏 국가")
    countries_options = ['all'] + st.session_state.countries
    country_filter = st.sidebar.selectbox(
        "국가 선택",
        options=countries_options,
        format_func=lambda x: '전체' if x == 'all' else x,
        help="특정 국가의 데이터만 조회하려면 선택하세요"
    )
    
    # 장르 선택
    st.sidebar.subheader("📚 장르")
    # 장르 매핑을 위한 딕셔너리
    genre_mapping = {
        "A": "환경·재난",
        "B": "미스터리·스릴러", 
        "C": "SF·판타지",
        "D": "사회·정치",
        "E": "이주·전쟁",
        "F": "젠더·다양성",
        "G": "종교·신화",
        "H": "관계·성장",
        "I": "로맨스",
        "J": "역사",
        "미분류": "기타"
    }
    
    genre_options = ['all'] + st.session_state.genre_codes
    genre_filter = st.sidebar.selectbox(
        "장르 선택",
        options=genre_options,
        format_func=lambda x: '전체' if x == 'all' else f"{x} ({genre_mapping.get(x, x)})",
        help="특정 장르의 데이터만 조회하려면 선택하세요"
    )
    
    
    st.sidebar.markdown("---")
    
    # 조회 버튼
    if st.sidebar.button("📊 데이터 조회", type="primary", use_container_width=True):
        with st.spinner("데이터를 조회하는 중..."):
            df = fetch_literature_data(
                start_date, end_date, original_filter, 
                country_filter, genre_filter,
            )
            
            if df is not None and not df.empty:
                # 세션 상태에 데이터 저장
                st.session_state.df_original = df
                st.session_state.data_loaded = True
                st.session_state.query_info = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'original_filter': original_filter,
                    'country_filter': country_filter,
                    'genre_filter': genre_filter,
                    # 'year_filter': year_filter,
                    'total_count': len(df)
                }
                st.success(f"총 {len(df)}건의 데이터를 조회했습니다.")
                
            elif df is not None and df.empty:
                st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
                st.session_state.data_loaded = False
            else:
                st.error("데이터 조회에 실패했습니다.")
                st.session_state.data_loaded = False
    
    # 현재 적용된 필터 조건 표시
    if st.session_state.data_loaded:
        st.subheader("🔍 적용된 필터 조건")
        query_info = st.session_state.query_info
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.metric("조회 기간", f"{query_info['start_date']} ~ {query_info['end_date']}")
        with col2:
            original_text = {'all': '전체', 'original': '원작', 'edition': '에디션'}[query_info['original_filter']]
            st.metric("원작여부", original_text)
        with col3:
            country_text = '전체' if query_info['country_filter'] == 'all' else query_info['country_filter']
            st.metric("국가", country_text)
        with col4:
            if query_info['genre_filter'] == 'all':
                genre_text = '전체'
            else:
                genre_text = f"{query_info['genre_filter']} ({genre_mapping.get(query_info['genre_filter'], query_info['genre_filter'])})"
            st.metric("장르", genre_text)
        
        # # 추가 정보
        # col5, col6 = st.columns(2)
        # with col5:
        #     # if query_info['year_filter'] == 'all':
        #     #     year_text = '전체'
        #     # elif isinstance(query_info['year_filter'], tuple):
        #     #     year_text = f"{query_info['year_filter'][0]} - {query_info['year_filter'][1]}"
        #     # else:
        #     #     year_text = str(query_info['year_filter'])
        #     # st.metric("연도", year_text)
        # with col6:
        #     st.metric("총 데이터 수", query_info['total_count'])
        
        st.markdown("---")
    
    # 데이터가 로드된 경우에만 표시
    if st.session_state.data_loaded and st.session_state.df_original is not None:
        df = st.session_state.df_original.copy()
        
        # 데이터 테이블 표시
        st.subheader("📋 조회 결과")
        
        # 페이지네이션을 위한 설정
        items_per_page = st.select_slider(
            "페이지당 항목 수", 
            options=[10, 25, 50, 100], 
            value=25,
            key="items_per_page"
        )
        
        total_pages = (len(df) - 1) // items_per_page + 1 if len(df) > 0 else 1
        page = st.number_input(
            f"페이지 (1-{total_pages})", 
            min_value=1, 
            max_value=max(1, total_pages), 
            value=1,
            key="current_page"
        )
        
        # 페이지별 데이터 표시
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_df = df.iloc[start_idx:end_idx]
        
        st.dataframe(
            page_df, 
            use_container_width=True,
            hide_index=True
        )
        
        # Excel 다운로드 버튼
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='문학도서데이터')
                
                # 워크시트 가져오기
                worksheet = writer.sheets['문학도서데이터']
                
                # 열 너비 자동 조정
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # 최대 50으로 제한
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            processed_data = output.getvalue()
            return processed_data
        
        excel_data = to_excel(df)
        query_info = st.session_state.query_info
        filename_parts = [
            f"literature_books_{query_info['start_date']}_{query_info['end_date']}",
            f"original_{query_info['original_filter']}" if query_info['original_filter'] != 'all' else '',
            f"country_{query_info['country_filter']}" if query_info['country_filter'] != 'all' else '',
            f"genre_{query_info['genre_filter']}" if query_info['genre_filter'] != 'all' else '',
            # f"year_{query_info['year_filter']}" if query_info['year_filter'] != 'all' else ''
        ]
        filename = "_".join([part for part in filename_parts if part]) + ".xlsx"
        
        st.download_button(
            label="📊 Excel 파일 다운로드",
            data=excel_data,
            file_name=filename,
            use_container_width=True , # 버튼 너비 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # 사용법 안내
    with st.expander("ℹ️ 사용법 안내"):
        st.markdown("""
        ### 📋 사용 순서
        1. **필터 조건 설정**: 좌측 사이드바에서 원하는 필터링 조건들을 설정하세요.
           - **기간**: 조회하고 싶은 발간일 범위
           - **원작여부**: 전체/원작/에디션 선택
           - **국가**: 특정 국가 선택 (선택사항)
           - **장르**: 특정 장르 선택 (선택사항)  
           - **연도**: 전체/범위/특정연도 선택 (선택사항)
           
        2. **데이터 조회**: 모든 조건 설정 후 '데이터 조회' 버튼 클릭
        3. **결과 확인**: 적용된 필터 조건과 조회 결과를 확인
        4. **다운로드**: 필요시 Excel 파일로 다운로드
        
        ### ⚠️ 유의사항
        - 기간 필터링은 발간일을 기준으로 적용됩니다
        - 여러 필터를 동시 적용하면 교집합 결과가 조회됩니다
        """)

if __name__ == "__main__":
    main()