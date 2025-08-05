import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime, date
from io import BytesIO

# 데이터베이스 연결 정보
DB_HOST = st.secrets["database"]["host"]
DB_NAME = st.secrets["database"]["database"]
DB_USER = st.secrets["database"]["user"]
DB_PASSWORD = st.secrets["database"]["password"]

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

def fetch_literature_data(start_date, end_date, original_filter='all'):
    """지정된 기간의 문학 도서 데이터 조회"""
    connection = get_database_connection()
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        # 원작여부 필터 조건 추가
        if original_filter == 'original':
            original_condition = "AND 원작여부 = 'original'"
        elif original_filter == 'edition':
            original_condition = "AND 원작여부 = 'edition'"
        else:  # 'all'
            original_condition = ""
        
        query = f"""
            SELECT id, year, 원작_제목, 에디션_제목, 작가명, `ISBN(13)`, ASIN, 
                유형, 출판사명, 언어, 발간일, 수집일자,  URL, 국가, 원작여부, genre1, genre2, genre3, genre4
            FROM literature_books 
            WHERE 발간일 BETWEEN %s AND %s
            {original_condition}
            ORDER BY 발간일 DESC
            """
            
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        
        # 컬럼명 가져오기
        column_names = [desc[0] for desc in cursor.description]
        
        # DataFrame 생성
        df = pd.DataFrame(results, columns=column_names)
        genre_columns = ['genre1', 'genre2', 'genre3', 'genre4']

        # 장르 코드 매핑 (예시)
        genre_mapping = {
                "A": "🌍 환경·재난",
                "B": "🔍 미스터리·스릴러", 
                "C": "🚀 SF·판타지",
                "D": "🏛️ 사회·정치",
                "E": "✈️ 이주·전쟁",
                "F": "🏳️‍🌈 젠더·다양성",
                "G": "⛪ 종교·신화",
                "H": "👩🏿‍🤝‍👨🏻 관계·성장",
                "I": "💕 로맨스",
                "J": "📜 역사",
                "미분류": "📚 기타"
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
                 '출판사명', '언어',  '장르', 'URL', '원작여부']]
        
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
        layout="wide",  # 이게 핵심!
        initial_sidebar_state="expanded"
            )
    st.title("📚 문학 도서 데이터 조회")
    st.markdown("---")
    
    # 🔐 비밀번호 입력
    secret_key_user = st.text_input(':closed_lock_with_key: **Secret Key**',
                                    placeholder='비밀번호를 입력해주세요.',
                                    type="password")
    
    # 비밀번호 확인
    if secret_key_user != st.secrets.get("app_password", "your_password"):
        st.warning("올바른 비밀번호를 입력해주세요.")
        st.stop()
    

    # 세션 상태 초기화
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'query_info' not in st.session_state:
        st.session_state.query_info = {}
    
    # 사이드바에 기간 설정
    st.sidebar.header("조회 조건 설정")
    # 원작여부 선택 추가
    original_filter = st.sidebar.selectbox(
        "원작여부 선택",
        options=['all', 'original', 'edition'],
        format_func=lambda x: {'all': '전체', 'original': '원작', 'edition': '에디션'}[x],
        help="조회할 데이터 유형을 선택하세요"
    )
    # 날짜 입력
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일", 
            value=date(2020, 1, 1),
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
    
    # 조회 버튼
    if st.sidebar.button("데이터 조회", type="primary"):
        with st.spinner("데이터를 조회하는 중..."):
            df = fetch_literature_data(start_date, end_date, original_filter) 
            
            if df is not None and not df.empty:
                # 세션 상태에 데이터 저장
                st.session_state.df_original = df
                st.session_state.data_loaded = True
                st.session_state.query_info = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'original_filter': original_filter,  # 필터 정보도 저장
                    'total_count': len(df)
                }
                st.success(f"총 {len(df)}건의 데이터를 조회했습니다.")
                
            elif df is not None and df.empty:
                st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
                st.session_state.data_loaded = False
            else:
                st.error("데이터 조회에 실패했습니다.")
                st.session_state.data_loaded = False
    
    # 데이터가 로드된 경우에만 표시
    if st.session_state.data_loaded and st.session_state.df_original is not None:
        df = st.session_state.df_original.copy()
        
        # 기본 정보 표시
        col1, col2, col3 = st.columns([3, 2, 1])  # 3:1:1 비율
        with col1:
            query_info = st.session_state.query_info
            st.metric("조회 기간", f"{query_info['start_date']} ~ {query_info['end_date']}")
        with col2:
            st.metric("총 데이터 수", query_info['total_count'])
        with col3:
            if not df.empty:
                unique_countries = df['국가'].nunique() if '국가' in df.columns else 0
                st.metric("국가 수", unique_countries)
        
        st.markdown("---")
        
        # 데이터 테이블 표시
        st.subheader("📋 조회 결과")
        
        # 데이터 필터링 옵션
        with st.expander("🔍 추가 필터링 옵션"):
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                if '국가' in df.columns:
                    countries = ['전체'] + sorted(df['국가'].dropna().unique().tolist())
                    selected_country = st.selectbox("국가 선택", countries)
                    if selected_country != '전체':
                        df = df[df['국가'] == selected_country]
            
            with filter_col2:
                # 장르 컬럼들을 합쳐서 유니크한 장르 목록 생성
                if any(col in df.columns for col in ['genre1', 'genre2', 'genre3', 'genre4']):
                    all_genres = set()
                    for genre_col in ['genre1', 'genre2', 'genre3', 'genre4']:
                        if genre_col in df.columns:
                            all_genres.update(df[genre_col].dropna().unique())
                    
                    genres = ['전체'] + sorted(list(all_genres))
                    selected_genre = st.selectbox("장르 선택", genres)
                    
                    if selected_genre != '전체':
                        # 4개 장르 컬럼 중 하나라도 선택한 장르와 일치하는 행 필터링
                        genre_mask = False
                        for genre_col in ['genre1', 'genre2', 'genre3', 'genre4']:
                            if genre_col in df.columns:
                                genre_mask = genre_mask | (df[genre_col] == selected_genre)
                        df = df[genre_mask]
        
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
        st.download_button(
            label="📊 Excel 파일 다운로드",
            data=excel_data,
            file_name=f"literature_books_{query_info['start_date']}_{query_info['end_date']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    
    # 사용법 안내
    with st.expander("ℹ️ 사용법 안내"):
        st.markdown("""
        1. **기간 설정**: 좌측 사이드바에서 조회하고 싶은 기간을 설정하세요.
        2. **데이터 조회**: '데이터 조회' 버튼을 클릭하여 데이터를 불러옵니다.
        3. **필터링**: 추가 필터링 옵션을 사용하여 데이터를 정렬할 수 있습니다.
        4. **다운로드**: 조회된 데이터를 Excel 파일로 다운로드할 수 있습니다.
        
        
        **주의사항**: 
        - 발간일을 기준으로 기간 필터링이 적용됩니다.
        """)



if __name__ == "__main__":
    main()