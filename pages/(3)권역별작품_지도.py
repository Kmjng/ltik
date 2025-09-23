import streamlit as st
import pydeck as pdk
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import base64 
from PIL import Image

logo = Image.open('./assets/logo1.jpg')  # 또는 'assets/logo.png'
def get_base64_image(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image('./assets/logo1.jpg')



# 페이지 설정을 wide로 변경
st.set_page_config(layout="wide")

# 전체 화면 스타일
st.markdown("""
<style>
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------- # 
# DB 연결 설정

DB_HOST = st.secrets["database"]["host"]
DB_NAME = st.secrets["database"]["database"]
DB_USER = st.secrets["database"]["user"]
DB_PASSWORD = st.secrets["database"]["password"]

# ------------------------------------------------------------------------- # 


def load_data_from_db(host, database, user, password, start_date=None, end_date=None, region=None):
    """DB에서 데이터 로드 및 전처리"""
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        
        if connection.is_connected():
            query = """
                SELECT 
                    lbr.book_id,
                    lbr.발간일,
                    lbr.권역,
                    lbr.국가,
                    lbr.원작여부,
                    lbr.원작_제목,
                    lbr.작가명,
                    gc.위도,
                    gc.경도,
                    CASE 
                        WHEN lbr.원작여부 = 'edition' THEN 
                            CASE 
                                WHEN lbr.국가 IN ('러시아', '카자흐스탄', '키르기스스탄', '타지키스탄', '투르크메니스탄', '우즈베키스탄') 
                                THEN '러시아-중앙아시아'
                                WHEN lbr.국가 IN ('스웨덴', '노르웨이', '핀란드', '덴마크', '아이슬란드') 
                                THEN '스칸디나비아'
                                WHEN gc.권역 = '중남미' THEN '라틴아메리카'
                                WHEN gc.권역 = '북미' THEN '북아메리카'  
                                WHEN gc.권역 = '아프리카' THEN '아프리카-중동'
                                WHEN gc.권역 = '중동' THEN '아프리카-중동'
                                ELSE gc.권역
                            END
                        ELSE 
                            CASE 
                                WHEN lbr.권역 = '중남미' THEN '라틴아메리카'
                                WHEN lbr.권역 = '북미' THEN '북아메리카'  
                                WHEN lbr.권역 = '아프리카' THEN '아프리카-중동'
                                WHEN lbr.권역 = '중동' THEN '아프리카-중동'
                                ELSE lbr.권역
                            END
                    END as 권역_x
                FROM literature_books_region lbr
                LEFT JOIN global_coor gc
                    ON lbr.국가 = gc.권역
                    OR lbr.국가 = gc.국가
                    OR lbr.국가 = gc.수도
            """
            
            where_conditions = []
            if start_date:
                where_conditions.append(f"lbr.발간일 >= '{start_date}'")
            if end_date:
                where_conditions.append(f"lbr.발간일 <= '{end_date}'")

            if region:  # 권역 필터 조건 추가 (매핑된 권역으로 비교)
                where_conditions.append(f"""(
                    CASE 
                        WHEN lbr.권역 = '중남미' THEN '라틴아메리카'
                        WHEN lbr.권역 = '북미' THEN '북아메리카'  
                        WHEN lbr.권역 = '아프리카' THEN '아프리카-중동'
                        WHEN lbr.권역 = '중동' THEN '아프리카-중동'
                        ELSE lbr.권역
                    END = '{region}'
                )""")

            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            # 필수 컬럼 확인
            required_cols = ['book_id', '발간일', '권역', '국가', '원작여부', '원작_제목', '작가명', '위도','경도','권역_x']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"누락된 컬럼: {missing_cols}")
                return pd.DataFrame()
            
            df['발간일'] = pd.to_datetime(df['발간일'])
            df = df.dropna(subset=required_cols)
            
            return df
            
    except Error as e:
        st.error(f"DB 연결 오류: {e}")
        return pd.DataFrame()




# 2. 권역 목록을 가져오는 함수 추가
def get_regions_from_db(host, database, user, password):
    """DB에서 출발지 권역 목록 가져오기"""
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        
        if connection.is_connected():
            query = "SELECT DISTINCT 권역 FROM literature_books_region WHERE 권역 IS NOT NULL ORDER BY 권역"
            df = pd.read_sql(query, connection)
            connection.close()
            return df['권역'].tolist()
    except Error as e:
        st.error(f"권역 목록 조회 오류: {e}")
        return []

def get_target_regions_from_db(host, database, user, password):
    """DB에서 도착지 권역 목록 가져오기 (권역_x 기준)"""
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        
        if connection.is_connected():
            query = """
                SELECT DISTINCT 
                    CASE 
                        WHEN lbr.국가 IN ('러시아', '카자흐스탄', '키르기스스탄', '타지키스탄', '투르크메니스탄', '우즈베키스탄') 
                        THEN '러시아-중앙아시아'
                        WHEN lbr.국가 IN ('스웨덴', '노르웨이', '핀란드', '덴마크', '아이슬란드') 
                        THEN '스칸디나비아'
                        WHEN gc.권역 = '중남미' THEN '라틴아메리카'
                        WHEN gc.권역 = '북미' THEN '북아메리카'  
                        WHEN gc.권역 = '아프리카' THEN '아프리카-중동'
                        WHEN gc.권역 = '중동' THEN '아프리카-중동'
                        ELSE gc.권역
                    END as 권역
                FROM literature_books_region lbr
                LEFT JOIN global_coor gc
                    ON lbr.국가 = gc.권역
                    OR lbr.국가 = gc.국가
                    OR lbr.국가 = gc.수도
                WHERE lbr.원작여부 = 'edition' AND (
                    gc.권역 IS NOT NULL 
                    OR lbr.국가 IN ('러시아', '카자흐스탄', '키르기스스탄', '타지키스탄', '투르크메니스탄', '우즈베키스탄')
                    OR lbr.국가 IN ('스웨덴', '노르웨이', '핀란드', '덴마크', '아이슬란드')
                )
                ORDER BY 권역
            """
            df = pd.read_sql(query, connection)
            connection.close()
            return df['권역'].tolist()
    except Error as e:
        st.error(f"도착지 권역 목록 조회 오류: {e}")
        return []



# 3. Streamlit UI 부분 - 권역 선택 추가

st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;">
            <h1>권역별 도서 진출 지도</h1>
        </div>
        """, unsafe_allow_html=True)
st.markdown("🗺️**권역별 도서 진출 지도를 표시합니다.**")
st.caption(f"*데이터 출처: Goodreads, GoogleSearch, Restcountries API*")

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

# 권역 목록 가져오기
regions_raw = get_regions_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
target_regions_raw = get_target_regions_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)

# 권역 매핑 설정 (합치고 싶은 권역들)
region_mapping = {
    '아프리카': '아프리카-중동',
    '중동': '아프리카-중동',
    '러시아': '러시아-중앙아시아',
    '중앙아시아': '러시아-중앙아시아',
    '중남미':'라틴아메리카', 
    '북미':'북아메리카'
}

# 매핑된 출발지 권역 목록 생성
regions = []
for region in regions_raw:
    mapped_region = region_mapping.get(region, region)
    if mapped_region not in regions:
        regions.append(mapped_region)

regions.sort()  # 정렬

# 매핑된 도착지 권역 목록 생성
target_regions = []
for region in target_regions_raw:
    mapped_region = region_mapping.get(region, region)
    if mapped_region not in target_regions:
        target_regions.append(mapped_region)

target_regions.sort()  # 정렬

# Streamlit 날짜 선택
start_date = st.date_input("🗓️**시작일**", value=datetime(1901, 1, 1),
               min_value=datetime(1901, 1, 1),  # 최소 날짜 설정
               max_value=datetime(2027, 12, 31) # 최대 날짜 설정
                           )
end_date = st.date_input("🗓️**종료일**", value=datetime.now(),
               min_value=datetime(1901, 1, 1),  # 최소 날짜 설정
               max_value=datetime(2027, 12, 31) # 최대 날짜 설정
                         )

# 출발지 권역 선택
selected_region = st.selectbox(
    "🌎 **출발지 권역 선택 (원작 기준)**",
    options=["선택해주세요"] + regions,
    index=0
)


# 도착지 권역 선택 (중복선택 가능) - 출발지와 동일한 권역을 디폴트로 설정
default_target_regions = []
if selected_region != "선택해주세요" and selected_region in target_regions:
    default_target_regions = [selected_region]

selected_target_regions = st.multiselect(
    "🌍**도착지 권역 선택 (번역 기준, 중복선택 가능)**",
    options=target_regions,
    default=default_target_regions
)


# 데이터 불러오기 - 출발지 권역이 선택된 경우에만
if selected_region != "선택해주세요":
    df = load_data_from_db(
        DB_HOST, DB_NAME, DB_USER, DB_PASSWORD,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        region=selected_region  # 선택된 출발지 권역 전달
    )
    
    # 도착지 권역이 선택된 경우 추가 필터링
    if selected_target_regions:
        # edition 데이터 중 선택된 도착지 권역에 해당하는 것만 필터링
        df = df[
            (df['원작여부'] == 'original') |  # 원작은 모두 포함
            ((df['원작여부'] == 'edition') & (df['권역_x'].isin(selected_target_regions)))  # 번역은 선택된 권역만
        ]
    

    if not df.empty:
        # st.dataframe(df)
        
        # 그래프 시각화 부분도 여기 안에 넣기
        df_raw = df.copy()
        
        # 데이터 가공 - 출발지(원작 권역) -> 도착지(번역 국가) 중복 제거 및 카운트
        originals = df_raw[df_raw['원작여부'] == 'original'].copy()
        editions = df_raw[df_raw['원작여부'] == 'edition'].copy()
        
        # 직접적인 경로 카운팅 방식으로 변경
        route_counts = {}
        
        # book_id별로 원작과 번역 정보 매칭
        for book_id in df_raw['book_id'].unique():
            book_data = df_raw[df_raw['book_id'] == book_id]
            orig_data = book_data[book_data['원작여부'] == 'original']
            ed_data = book_data[book_data['원작여부'] == 'edition']
            
            # 하나의 책에서 원작과 번역이 모두 있는 경우에만 경로 생성
            if len(orig_data) > 0 and len(ed_data) > 0:
                for _, orig_row in orig_data.iterrows():
                    for _, ed_row in ed_data.iterrows():
                        route_key = f"{orig_row['국가']} → {ed_row['국가']}"
                        
                        # 좌표 정보와 함께 저장
                        route_info = {
                            'src_country': orig_row['국가'],
                            'src_region': orig_row['권역'],
                            'src_lat': orig_row['위도'],
                            'src_lon': orig_row['경도'],
                            'tgt_country': ed_row['국가'],
                            'tgt_region_x': ed_row['권역_x'],  # edition 권역 
                            'tgt_lat': ed_row['위도'],
                            'tgt_lon': ed_row['경도'],
                            'route': route_key
                        }
                        
                        # 같은 경로 카운트 증가
                        if route_key in route_counts:
                            route_counts[route_key]['count'] += 1
                        else:
                            route_counts[route_key] = {**route_info, 'count': 1}
        
        if route_counts:  # 연결할 데이터가 있는 경우에만
            merged_df = pd.DataFrame(list(route_counts.values()))
            
            # 번역 횟수가 1회 이상인 경로만 표시 (필요시 이 값을 높여서 선의 개수 조절)
            merged_df = merged_df[merged_df['count'] >= 1]
            
            # 양방향 경로 체크 및 높이 설정
            merged_df['height'] = 0.3  # 기본 높이
            
            # 양방향 경로가 있는지 확인하고 높이 조정
            for i, row in merged_df.iterrows():
                src_country = row['src_country']
                tgt_country = row['tgt_country']
                reverse_route = f"{tgt_country} → {src_country}"
                
                # 역방향 경로가 존재하는지 확인
                reverse_exists = any(merged_df['route'] == reverse_route)
                
                if reverse_exists:
                    # 알파벳 순으로 정렬했을 때 먼저 오는 경로는 높이 0.15, 나중 오는 경로는 0.45
                    if src_country < tgt_country:
                        merged_df.at[i, 'height'] = 0.15  # 낮은 높이
                    else:
                        merged_df.at[i, 'height'] = 0.35  # 높은 높이
            
            # 출발지 권역과 도착지 권역 비교하여 색상 구분 컬럼 추가
            merged_df['is_same_region'] = merged_df['tgt_region_x'] == selected_region
            
            # 색상 설정 (동일한 권역: 주황-파랑, 다른 권역: 연두-연두)
            merged_df['source_color'] = merged_df['is_same_region'].apply(
                lambda x: [255, 140, 0, 160] if x else [144, 238, 144, 160]  # 주황 vs 연두
            )
            merged_df['target_color'] = merged_df['is_same_region'].apply(
                lambda x: [0, 128, 255, 160] if x else [144, 238, 144, 160]   # 파랑 vs 연두
            )
            
            # 번역 횟수 기준으로 정렬
            merged_df = merged_df.sort_values('count', ascending=False)
            
            # 상위 20개 경로만 표시하여 선의 개수 제한 (필요시 조절 가능)
            # top_routes = merged_df.head(20)
            top_routes = merged_df.copy()

           
            # 전체 경로 통계 표시
            filter_info = f"출발지: {selected_region}"
            if selected_target_regions:
                filter_info += f", 도착지: {', '.join(selected_target_regions)}"
            # st.info(f"📈 {filter_info} 조건에서 총 {len(merged_df)}개의 경로를 지도에 표시합니다. (총 번역 횟수: {merged_df['count'].sum()}회)")
            
            
            # 색상 구분 안내
            same_region_count = len(merged_df[merged_df['is_same_region']])
            different_region_count = len(merged_df[~merged_df['is_same_region']])
            if different_region_count > 0:
                st.info(f"📈 동일 권역 경로 {same_region_count}개(🟠→🔵), 타 권역 경로 {different_region_count}개(🟢→🟣)")
            else: 
                st.info(f"📈 {filter_info} 조건에서 총 {len(merged_df)}개의 경로를 지도에 표시합니다(🟠→🔵).")


            # 선 두께 1로 고정 (width 컬럼 계산 불필요)
            
            # 권역별 중심 좌표 매핑
            region_coordinates = {
                '동아시아': {'lat': 35, 'lon': 105, 'zoom': 3.5},
                '동남아시아': {'lat': 10, 'lon': 110, 'zoom': 4},
                '남아시아': {'lat': 20, 'lon': 78, 'zoom': 4},
                '서아시아': {'lat': 29, 'lon': 53, 'zoom': 4},
                '유럽': {'lat': 54, 'lon': 15, 'zoom': 3.5},
                '스칸디나비아': {'lat': 64, 'lon': 26, 'zoom': 4},
                '러시아-중앙아시아': {'lat': 55, 'lon': 90, 'zoom': 3},
                '북아메리카': {'lat': 45, 'lon': -100, 'zoom': 3},
                '라틴아메리카': {'lat': -15, 'lon': -60, 'zoom': 3},
                '아프리카-중동': {'lat': 0, 'lon': 20, 'zoom': 3},
                '오세아니아': {'lat': -25, 'lon': 140, 'zoom': 4}
            }
            
            # 선택된 권역에 따른 뷰 설정
            if selected_region in region_coordinates:
                coord = region_coordinates[selected_region]
                view_lat, view_lon, view_zoom = coord['lat'], coord['lon'], coord['zoom']
            else:
                # 기본 전체 지도 뷰
                view_lat, view_lon, view_zoom = 35, 50, 2.5
            
            # Pydeck 시각화 코드 (상위 경로만 표시, 선 두께 고정)
            layer = pdk.Layer(
                "ArcLayer",
                data=top_routes,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["tgt_lon", "tgt_lat"],
                get_source_color="source_color",  # 동적 색상 컬럼 사용
                get_target_color="target_color",  # 동적 색상 컬럼 사용
                # get_width="width",  # 번역 횟수에 따른 선 두께
                get_width=3.5,  
                get_height="height",  # 양방향 경로 구분을 위한 동적 높이
                pickable=True,
                auto_highlight=True,
            )
            
            view_state = pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=view_zoom, pitch=30)
            
            tooltip = {
                "html": """
                <b>경로:</b> {route}<br/>
                <b>출발지:</b> {src_country}<br/>
                <b>도착지:</b> {tgt_country}<br/>
                <b>번역 횟수:</b> {count}회<br/>
                """,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "border": "1px solid white",
                    "borderRadius": "5px",
                    "padding": "10px"
                }
            }
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/light-v9',
                tooltip=tooltip
            )
            
            st.pydeck_chart(r, use_container_width=True, height=1200)

            # ------------------------------------------------------ # 
            # 카운트 정보를 보여주는 테이블 추가
            st.subheader("📊 경로별 번역 횟수")
            display_df = top_routes[['src_region', 'route', 'tgt_region_x', 'count']].copy()
            display_df.columns = ['출발지(원작 권역)', '경로', '도착지(번역 권역)', '번역 횟수']
            st.dataframe(display_df, use_container_width=True)
            
            # 실제 책 원작 정보 테이블 추가
            st.subheader("📚 원작 작품 목록")
            
            # 현재 조건에 맞는 원작 데이터 추출
            original_books = df_raw[df_raw['원작여부'] == 'original'].copy()
            
            # 번역된 책들만 필터링 (route_counts에 있는 book_id만)
            translated_book_ids = []
            for book_id in df_raw['book_id'].unique():
                book_data = df_raw[df_raw['book_id'] == book_id]
                orig_data = book_data[book_data['원작여부'] == 'original']
                ed_data = book_data[book_data['원작여부'] == 'edition']
                if len(orig_data) > 0 and len(ed_data) > 0:
                    translated_book_ids.append(book_id)
            
            original_books_filtered = original_books[original_books['book_id'].isin(translated_book_ids)]
            
            if not original_books_filtered.empty:
                # 각 원작에 대한 번역 국가 정보 추가
                books_with_translations = []
                for _, orig_row in original_books_filtered.iterrows():
                    book_id = orig_row['book_id']
                    # 해당 책의 번역 국가들 찾기 (날짜 순으로 정렬)
                    translations = df_raw[(df_raw['book_id'] == book_id) & (df_raw['원작여부'] == 'edition')].sort_values('발간일')
                    translation_countries = translations['국가'].tolist()  # unique() 대신 tolist()로 순서 유지
                    translation_regions = translations['권역_x'].tolist()
                    
                    # 중복 제거하되 순서는 유지 (첫 번째 번역 날짜 기준)
                    seen_countries = set()
                    ordered_countries = []
                    for country in translation_countries:
                        if country not in seen_countries:
                            ordered_countries.append(country)
                            seen_countries.add(country)
                    
                    seen_regions = set()
                    ordered_regions = []
                    for region in translation_regions:
                        if pd.notna(region) and region not in seen_regions:
                            ordered_regions.append(str(region))
                            seen_regions.add(region)
                    
                    books_with_translations.append({
                        '원작_제목': orig_row['원작_제목'],
                        '작가': orig_row['작가명'],
                        '발간일': orig_row['발간일'].strftime('%Y-%m-%d'),
                        '원작_권역': orig_row['권역'],
                        '원작_국가': orig_row['국가'],
                        '번역_국가들': ' → '.join(ordered_countries),  # 화살표로 연결
                        '번역_권역들': ', '.join(ordered_regions),      # 콤마로 구분
                        '번역_횟수': len(translations)
                    })
                
                books_df = pd.DataFrame(books_with_translations)
                books_df.columns = ['작품명', '작가명', '발간일', '원작 권역', '원작 국가', '번역된 국가들', '번역된 권역들', '번역 횟수']
                
                # 발간일 순으로 정렬
                books_df = books_df.sort_values('발간일')
                st.info(f"📖 총 {len(books_df)}권의 원작이 있습니다.")
                st.dataframe(books_df, use_container_width=True)
                
            else:
                st.warning("조건에 맞는 번역된 원작이 없습니다.")
            
            # ------------------------------------------------------ # 

        else:
            if selected_target_regions:
                st.info(f"출발지 권역({selected_region})에서 도착지 권역({', '.join(selected_target_regions)})으로의 원작과 번역을 연결할 수 있는 데이터가 없습니다.")
            else:
                st.info(f"출발지 권역({selected_region})에서 원작과 번역을 연결할 수 있는 데이터가 없습니다.")
    else:
        if selected_target_regions:
            st.warning(f"출발지 권역({selected_region})에서 도착지 권역({', '.join(selected_target_regions)})으로의 조건에 해당하는 데이터가 없습니다.")
        else:
            st.warning(f"출발지 권역({selected_region})에 해당하는 데이터가 없습니다.")
else:
    st.info("출발지 권역을 선택하면 데이터와 시각화가 표시됩니다. 도착지 권역 선택은 선택사항입니다.")








# 'mapbox://styles/mapbox/light-v9' - 밝은 배경
# 'mapbox://styles/mapbox/dark-v9' - 어두운 배경
# 'mapbox://styles/mapbox/satellite-v9' - 위성 이미지
# 'mapbox://styles/mapbox/streets-v11' - 일반 도로지도
# 'mapbox://styles/mapbox/outdoors-v11' - 야외활동용 지도


