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

def fetch_literature_data(start_date, end_date):
    """지정된 기간의 문학 도서 데이터 조회"""
    connection = get_database_connection()
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        query = """
        SELECT id, year, 원작_제목, 에디션_제목, 작가명, `ISBN(13)`, ASIN, 
               유형, 출판사명, 언어, 발간일, 수집일자,  URL, 국가, 원작여부
        FROM literature_books 
        WHERE 원작여부 = 'original' 
        AND 발간일 BETWEEN %s AND %s
        ORDER BY 발간일 DESC
        """
        
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        
        # 컬럼명 가져오기
        column_names = [desc[0] for desc in cursor.description]
        
        # DataFrame 생성
        df = pd.DataFrame(results, columns=column_names)
        
        return df
        
    except Error as e:
        st.error(f"데이터 조회 오류: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def main():
    st.title("📚 문학 도서 데이터 조회")
    st.markdown("---")
    
    # 세션 상태 초기화
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'query_info' not in st.session_state:
        st.session_state.query_info = {}
    
    # 사이드바에 기간 설정
    st.sidebar.header("조회 조건 설정")
    
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
            df = fetch_literature_data(start_date, end_date)
            
            if df is not None and not df.empty:
                # 세션 상태에 데이터 저장
                st.session_state.df_original = df
                st.session_state.data_loaded = True
                st.session_state.query_info = {
                    'start_date': start_date,
                    'end_date': end_date,
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
        col1, col2, col3 = st.columns(3)
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
                if '유형' in df.columns:
                    types = ['전체'] + sorted(df['유형'].dropna().unique().tolist())
                    selected_type = st.selectbox("유형 선택", types)
                    if selected_type != '전체':
                        df = df[df['유형'] == selected_type]
        
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
        3. **필터링**: 추가 필터링 옵션을 사용하여 국가나 유형별로 데이터를 정렬할 수 있습니다.
        4. **다운로드**: 조회된 데이터를 Excel 파일로 다운로드할 수 있습니다.
        
        
        **주의사항**: 
        - 원작여부가 'original'인 데이터만 조회됩니다.
        - 발간일을 기준으로 기간 필터링이 적용됩니다.
        """)



if __name__ == "__main__":
    main()